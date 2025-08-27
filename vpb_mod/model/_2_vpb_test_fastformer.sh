#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ================== CONFIG CỐ ĐỊNH (sửa nếu cần) ==================
PYMOD="vpb_mod.model._2_fastformer_infer"
BASE_CONFIG="tutorials/asr/configs/fast-conformer_transducer_bpe.yaml"
DEVICES="1"
PRECISION="16"
BATCH_SIZE="64"
EXP_DIR="../nemo_work/_1_small_vi_ds/experiments"

# ================== ROOT PATHS ==================
MANIFEST_ROOT="/home/ubuntu/work/clean_dataset_vpb/manifest"

# ================== DANH SÁCH DATASETS VPB (đã convert _nemo.jsonl) ==================
declare -A TESTSETS=(
  ["standard_test_2"]="${MANIFEST_ROOT}/standard_test_2/test_meta_nemo.jsonl"
  ["standard_test"]="${MANIFEST_ROOT}/standard_test/test_meta_nemo.jsonl"
  ["next_day_test_debug"]="${MANIFEST_ROOT}/standard_test/next_day_test_meta_debug_nemo.jsonl"
  ["vpb_right2_train"]="${MANIFEST_ROOT}/manifest_vpb_right_2/train_meta_nemo.jsonl"
  ["vpb_right2_valid"]="${MANIFEST_ROOT}/manifest_vpb_right_2/valid_meta_nemo.jsonl"
)

# ================== DANH SÁCH MODEL .NEMO CẦN TEST ==================
MODELS=(
  "../nemo_work/_1_small_vi_ds/experiments/vietspeech/vpb_asr_fastconformer/2025-08-25_07-42-00/checkpoints/vpb_asr_fastconformer.nemo"
  # "../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer.nemo"
)

# ================== OUTPUT ==================
STAMP="$(date +'%Y%m%d_%H%M%S')"
OUT_ROOT="nemo_eval_${STAMP}"
LOG_DIR="${OUT_ROOT}/logs"
SUMMARY_TSV="${OUT_ROOT}/vpb_mod/logs/summary.tsv"

mkdir -p "${LOG_DIR}" "$(dirname "${SUMMARY_TSV}")"
# Header (TSV)
printf "model_tag\tdataset\twer\tlog_path\n" > "${SUMMARY_TSV}"

# ================== HÀM TRỢ GIÚP ==================
normpath() {
  python3 - <<'PY'
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
}

model_tag_from_path() {
  local nemo="$1"
  local nemo_base="$(basename "${nemo}")"
  local nemo_name="${nemo_base%.nemo}"
  local ts_dir="$(basename "$(dirname "$(dirname "$(normpath "${nemo}")")")")" 2>/dev/null || true
  if [[ "${ts_dir}" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "${ts_dir}__${nemo_name}"
  else
    echo "${nemo_name}"
  fi
}

extract_wer() {
  local logfile="$1"
  awk '
    /Final WER for the test set:/ {
      for(i=1;i<=NF;i++){
        if ($i ~ /^[0-9]+\.[0-9]+$/) last=$i
      }
    }
    END { if (last!="") print last; }
  ' "${logfile}" || true
}

# ================== CHẠY TOÀN BỘ ==================
echo "==> Bắt đầu batch eval. Output: ${OUT_ROOT}"
for nemo in "${MODELS[@]}"; do
  if [[ ! -f "${nemo}" ]]; then
    echo "!! WARNING: Không tìm thấy model file: ${nemo}, bỏ qua."
    continue
  fi

  model_tag="$(model_tag_from_path "${nemo}")"
  echo "-- Model: ${model_tag}"

  for ds in "${!TESTSETS[@]}"; do
    manifest="${TESTSETS[$ds]}"
    if [[ ! -f "${manifest}" ]]; then
      echo "!! WARNING: Không thấy manifest: ${manifest}, bỏ qua dataset ${ds}."
      continue
    fi

    run_name="testonly_${ds}__${model_tag}"
    log_path="${LOG_DIR}/${run_name}.log"

    echo "   -> Dataset: ${ds}"
    echo "      Log: ${log_path}"

    EXP_NAME="vpb_asr_fastconformer_testonly_${ds}__${model_tag}"

    set +e
    python -m "${PYMOD}" \
      --base-config "${BASE_CONFIG}" \
      --test-manifest "${manifest}" \
      --devices "${DEVICES}" \
      --precision "${PRECISION}" \
      --batch-size "${BATCH_SIZE}" \
      --exp-dir "${EXP_DIR}" \
      --exp-name "${EXP_NAME}" \
      --nemo "${nemo}" \
      --test-only 2>&1 | tee "${log_path}"
    rc=$?
    set -e

    wer_val="$(extract_wer "${log_path}")"
    [[ -z "${wer_val}" ]] && wer_val="NA"

    abs_log="$(normpath "${log_path}")"
    printf "%s\t%s\t%s\t%s\n" "${model_tag}" "${ds}" "${wer_val}" "${abs_log}" >> "${SUMMARY_TSV}"

    if [[ ${rc} -ne 0 ]]; then
      echo "      >> RUN ERROR (rc=${rc}), vẫn tiếp tục testcase khác."
    else
      echo "      >> DONE: WER=${wer_val}"
    fi
  done
done

echo
echo "================= TÓM TẮT KẾT QUẢ (TSV) ================="
column -s $'\t' -t < "${SUMMARY_TSV}" | sed '1q;2,$s/^/  /'
echo "========================================================="
echo "Summary TSV: ${SUMMARY_TSV}"
echo "Logs folder: ${LOG_DIR}"
