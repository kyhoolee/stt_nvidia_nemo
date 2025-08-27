#!/usr/bin/env bash
set -euo pipefail
SITE=/etc/nginx/sites-available/streamlit_8501
LINK=/etc/nginx/sites-enabled/streamlit_8501

# 1) Tạo server block làm default_server, proxy tới 127.0.0.1:8501
sudo tee "$SITE" >/dev/null <<'EOF'
server {
    listen 80 default_server;
    server_name _;

    location / {
        proxy_pass         http://127.0.0.1:8501/;
        proxy_http_version 1.1;

        # WebSocket (Streamlit cần)
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";

        # Forward headers
        proxy_set_header   Host $host;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        proxy_read_timeout 3600;
        proxy_send_timeout 3600;
    }
}
EOF

# 2) Tắt trang mặc định nếu còn
sudo rm -f /etc/nginx/sites-enabled/default

# 3) Bật site proxy
sudo ln -sf "$SITE" "$LINK"

# 4) Kiểm tra & reload
sudo nginx -t
sudo systemctl reload nginx

# 5) In trạng thái nhanh
echo; echo "Active servers:"
sudo nginx -T | sed -n '1,160p' | grep -E 'server \{|listen 80|server_name' -n || true
echo; echo "Try:  curl -I http://127.0.0.1  &&  curl -I http://<PUBLIC_IP>"
