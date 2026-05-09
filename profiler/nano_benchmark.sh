#!/bin/bash
# ============================================================
# nano-dist-spec speculative decoding benchmark script
# 依次跑: baseline（单独 basic）+ spec（仅吞吐与投机指标；加速比请用 baseline JSON 自行对比）
# ============================================================

set -u

# ---------- 可配置参数 ----------
TARGET_MODEL="/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DRAFT_MODEL="/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

INPUT_LEN=128
OUTPUT_LEN=256
NUM_PROMPTS=5
MAX_NUM_SEQS=1
MAX_MODEL_LEN=2048
TP_SIZE=1
NUM_GPU_BLOCKS=128

# k 值扫描范围
K_VALUES=(1 2 3 4 5 6 7)

# bench 脚本路径（默认按仓库根目录执行）
BENCH_PY="profiler/bench.py"

# 日志目录（按时间戳区分每次运行）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./nano_spec_bench_logs_${TIMESTAMP}"
OUT_DIR="${LOG_DIR}/bench_results"
SUMMARY_LOG="${LOG_DIR}/summary.log"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# ---------- 工具函数 ----------
log_summary() {
    echo "$1" | tee -a "${SUMMARY_LOG}"
}

latest_json_file() {
    local mode="$1"
    python - "$OUT_DIR" "$mode" <<'PY'
import glob
import os
import sys

out_dir = sys.argv[1]
mode = sys.argv[2]
files = glob.glob(os.path.join(out_dir, f"{mode}_*.json"))
if not files:
    print("")
else:
    print(max(files, key=os.path.getmtime))
PY
}

extract_basic_metrics() {
    local json_file="$1"
    python - "$json_file" <<'PY'
import json
import sys

path = sys.argv[1]
if not path:
    print("Throughput: <missing json file>")
    raise SystemExit(0)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

tp = data.get("summary", {}).get("throughput", {})
rps = tp.get("request_throughput_rps", 0.0)
tps = tp.get("output_token_throughput_tps", 0.0)
elapsed = tp.get("elapsed_s", 0.0)
print(
    "Throughput: "
    f"request={rps:.4f} req/s, "
    f"output={tps:.4f} tok/s, "
    f"elapsed={elapsed:.4f}s"
)
PY
}

extract_spec_metrics() {
    local json_file="$1"
    local k="$2"
    python - "$json_file" "$k" <<'PY'
import json
import sys

path = sys.argv[1]
k = int(sys.argv[2])

if not path:
    print("Throughput: <missing json file>")
    print("SpecDecoding: <missing json file>")
    raise SystemExit(0)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

entries = data.get("sweep", [])
row = None
for item in entries:
    if int(item.get("K", -1)) == k:
        row = item
        break

if row is None:
    print("Throughput: <missing sweep row>")
    print("SpecDecoding: <missing sweep row>")
    raise SystemExit(0)

tp = row.get("throughput", {})
rps = tp.get("request_throughput_rps", 0.0)
tps = tp.get("output_token_throughput_tps", 0.0)
elapsed = tp.get("elapsed_s", 0.0)
accept = row.get("draft_accept_rate", 0.0)
tokens_per_round = row.get("tokens_per_round", 0.0)

print(
    "Throughput: "
    f"request={rps:.4f} req/s, "
    f"output={tps:.4f} tok/s, "
    f"elapsed={elapsed:.4f}s"
)
print(
    "SpecDecoding: "
    f"accept_rate={accept:.4%}, "
    f"tokens_per_round={tokens_per_round:.4f}"
)
PY
}

# ---------- 开场打印 ----------
log_summary "============================================================"
log_summary "Nano Spec Decoding Benchmark - ${TIMESTAMP}"
log_summary "Target: ${TARGET_MODEL}"
log_summary "Draft:  ${DRAFT_MODEL}"
log_summary "input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}, num_prompts=${NUM_PROMPTS}"
log_summary "max_num_seqs=${MAX_NUM_SEQS}, tp=${TP_SIZE}, max_model_len=${MAX_MODEL_LEN}"
log_summary "num_gpu_blocks=${NUM_GPU_BLOCKS}"
log_summary "log_dir=${LOG_DIR}"
log_summary "============================================================"

# ---------- 1. Baseline ----------
BASELINE_LOG="${LOG_DIR}/baseline.log"
log_summary ""
log_summary ">>> [1/$((${#K_VALUES[@]} + 1))] Running BASELINE (no spec decode)..."
log_summary "Start time: $(date +"%Y-%m-%d %H:%M:%S")"

python "${BENCH_PY}" --out-dir "${OUT_DIR}" basic \
    --model "${TARGET_MODEL}" \
    --input-len "${INPUT_LEN}" \
    --output-len "${OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --num-gpu-blocks "${NUM_GPU_BLOCKS}" \
    > "${BASELINE_LOG}" 2>&1

BASELINE_EXIT=$?
BASELINE_JSON=$(latest_json_file "basic")
log_summary "End time:   $(date +"%Y-%m-%d %H:%M:%S")"
log_summary "Exit code:  ${BASELINE_EXIT}"
log_summary "Result json: ${BASELINE_JSON:-N/A}"
log_summary "========== [BASELINE] =========="
extract_basic_metrics "${BASELINE_JSON}" | tee -a "${SUMMARY_LOG}"
log_summary "Log file: ${BASELINE_LOG}"

# ---------- 2. Spec decoding with k=1..7 ----------
IDX=2
for K in "${K_VALUES[@]}"; do
    SPEC_LOG="${LOG_DIR}/spec_k${K}.log"
    log_summary ""
    log_summary ">>> [${IDX}/$((${#K_VALUES[@]} + 1))] Running SPEC DECODE k=${K}..."
    log_summary "Start time: $(date +"%Y-%m-%d %H:%M:%S")"

    python "${BENCH_PY}" --out-dir "${OUT_DIR}" spec \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --input-len "${INPUT_LEN}" \
        --output-len "${OUTPUT_LEN}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --k-values "${K}" \
        --num-gpu-blocks "${NUM_GPU_BLOCKS}" \
        > "${SPEC_LOG}" 2>&1

    SPEC_EXIT=$?
    SPEC_JSON=$(latest_json_file "spec")
    log_summary "End time:   $(date +"%Y-%m-%d %H:%M:%S")"
    log_summary "Exit code:  ${SPEC_EXIT}"
    log_summary "Result json: ${SPEC_JSON:-N/A}"
    log_summary "========== [SPEC k=${K}] =========="
    extract_spec_metrics "${SPEC_JSON}" "${K}" | tee -a "${SUMMARY_LOG}"
    log_summary "Log file: ${SPEC_LOG}"

    IDX=$((IDX + 1))
    sleep 5
done

# ---------- 最终汇总 ----------
log_summary ""
log_summary "============================================================"
log_summary "ALL DONE. Full logs saved under: ${LOG_DIR}"
log_summary "Summary log: ${SUMMARY_LOG}"
log_summary "============================================================"

echo ""
echo "=============== Final Throughput Comparison ==============="
python - "$SUMMARY_LOG" <<'PY'
import re
import sys

summary_log = sys.argv[1]
rows = []
current_tag = None
with open(summary_log, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        m = re.match(r"========== \[(.+)\] ==========$", line)
        if m:
            current_tag = m.group(1)
            continue
        if line.startswith("Throughput:") and current_tag is not None:
            rows.append((current_tag, line))
            current_tag = None

print("Config       | Throughput line")
print("-------------|---------------------------------------------------------------")
for tag, tp in rows:
    print(f"{tag:<12} | {tp}")
PY
