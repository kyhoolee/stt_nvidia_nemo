#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG
########################################
SERVER_NAME="_"               # dùng "_" cho mọi host, hoặc đặt domain nếu muốn (không bật HTTPS)
BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8501"

CONFIGURE_UFW=true            # mở port 80 (HTTP) nếu có UFW
ENABLE_BASIC_AUTH=false       # bật Basic Auth nếu muốn
BASIC_AUTH_USER="admin"
BASIC_AUTH_FILE="/etc/nginx/.htpasswd"

# Allowlist IP (để trống = cho tất cả). Ví dụ: ("1.2.3.4" "5.6.7.8")
ALLOW_IPS=()

########################################
log() { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERR ]\033[0m $*" >&2; }

need_root() {
  if [[ $EUID -ne 0 ]]; then
    err "Vui lòng chạy bằng sudo/root: sudo $0"
    exit 1
  fi
}

detect_cmd() { command -v "$1" >/dev/null 2>&1; }

ensure_pkg() {
  log "Cài đặt gói (nếu thiếu): $*"
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
}

setup_ufw() {
  if ! detect_cmd ufw; then
    warn "UFW chưa cài, bỏ qua cấu hình firewall bằng UFW."
    return
  fi
  log "Mở UFW cổng 80/tcp (HTTP)"
  ufw allow 80/tcp || true
  if ufw status | grep -q "Status: inactive"; then
    warn "UFW đang INACTIVE. Bật UFW? (y/N)"
    read -r ans
    if [[ "${ans,,}" == "y" ]]; then ufw enable; else warn "Để nguyên UFW inactive."; fi
  fi
}

gen_allowlist_block() {
  local block=""
  if (( ${#ALLOW_IPS[@]} > 0 )); then
    block+="\n    # Allowlist IP\n"
    block+="    allow 127.0.0.1;\n"
    for ip in "${ALLOW_IPS[@]}"; do
      block+="    allow ${ip};\n"
    done
    block+="    deny all;\n"
  fi
  echo -e "$block"
}

main() {
  need_root
  ensure_pkg nginx

  local AUTH_SNIPPET=""
  if [[ "$ENABLE_BASIC_AUTH" == "true" ]]; then
    ensure_pkg apache2-utils
    if [[ ! -f "$BASIC_AUTH_FILE" ]]; then
      log "Tạo Basic Auth file: $BASIC_AUTH_FILE (sẽ hỏi password cho user $BASIC_AUTH_USER)"
      htpasswd -c "$BASIC_AUTH_FILE" "$BASIC_AUTH_USER"
    else
      log "Đã có $BASIC_AUTH_FILE, bỏ qua tạo mới."
    fi
    AUTH_SNIPPET=$(
      cat <<'EOF'
        auth_basic "Restricted";
        auth_basic_user_file /etc/nginx/.htpasswd;
EOF
    )
  fi

  local ALLOWLIST_SNIPPET
  ALLOWLIST_SNIPPET="$(gen_allowlist_block)"

  local SITE_FILE="/etc/nginx/sites-available/streamlit_8501"
  local SITE_LINK="/etc/nginx/sites-enabled/streamlit_8501"

  log "Tạo Nginx server block: $SITE_FILE"
  cat > "$SITE_FILE" <<EOF
server {
    listen 80;
    server_name ${SERVER_NAME};

    location / {
        proxy_pass         http://${BACKEND_HOST}:${BACKEND_PORT}/;
        proxy_http_version 1.1;

        # WebSocket
        proxy_set_header   Upgrade \$http_upgrade;
        proxy_set_header   Connection "upgrade";

        # Forward headers
        proxy_set_header   Host \$host;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;

        proxy_read_timeout 3600;
        proxy_send_timeout 3600;
${AUTH_SNIPPET}
${ALLOWLIST_SNIPPET}
    }
}
EOF

  ln -sf "$SITE_FILE" "$SITE_LINK"
  nginx -t
  systemctl reload nginx

  if [[ "$CONFIGURE_UFW" == "true" ]]; then
    setup_ufw
  fi

  echo
  log "XONG. App nội bộ: http://${BACKEND_HOST}:${BACKEND_PORT}"
  echo "Truy cập ngoài:  http://<PUBLIC_IP>/  (hoặc http://<domain nếu đã trỏ DNS>)"
  echo "Kiểm tra Nginx:  sudo tail -f /var/log/nginx/error.log"
}

main "$@"
