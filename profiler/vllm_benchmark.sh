#!/bin/bash
# ============================================================
# DeepSeek-R1-Distill-Qwen 32B + 7B 投机解码 benchmark 脚本
# 依次跑: baseline + k=1,2,3,4,5,6,7
# ============================================================

set -u  # 未定义变量时报错（不用 -e，避免单轮失败中断全部）

# ---------- 可配置参数 ----------
TARGET_MODEL="/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DRAFT_MODEL="/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

INPUT_LEN=128
OUTPUT_LEN=256
NUM_PROMPTS=50
MAX_NUM_SEQS=1
MAX_MODEL_LEN=2048
GPU_MEM_UTIL=0.92
TP_SIZE=1

# k 值扫描范围
K_VALUES=(1 2 3 4 5 6 7)

# 日志目录（按时间戳区分每次运行）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./spec_bench_logs_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/summary.log"

# ---------- 工具函数 ----------
log_summary() {
    echo "$1" | tee -a "${SUMMARY_LOG}"
}

# 从单次运行日志里提取关键指标写入 summary
extract_metrics() {
    local tag="$1"
    local log_file="$2"
    log_summary ""
    log_summary "========== [${tag}] =========="

    # Throughput 行
    local throughput_line
    throughput_line=$(grep -E "Throughput:" "${log_file}" | tail -n 1)
    log_summary "Throughput: ${throughput_line:-N/A}"

    # SpecDecoding metrics（如果有）
    local spec_line
    spec_line=$(grep -E "SpecDecoding metrics" "${log_file}" | tail -n 1)
    if [[ -n "${spec_line}" ]]; then
        log_summary "SpecDecoding: ${spec_line}"
    fi

    log_summary "Log file: ${log_file}"
}

# ---------- 开场打印 ----------
log_summary "============================================================"
log_summary "Spec Decoding Benchmark - ${TIMESTAMP}"
log_summary "Target: ${TARGET_MODEL}"
log_summary "Draft:  ${DRAFT_MODEL}"
log_summary "input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}, num_prompts=${NUM_PROMPTS}"
log_summary "max_num_seqs=${MAX_NUM_SEQS}, tp=${TP_SIZE}, gpu_mem=${GPU_MEM_UTIL}"
log_summary "============================================================"

# ---------- 1. Baseline ----------
BASELINE_LOG="${LOG_DIR}/baseline.log"
log_summary ""
log_summary ">>> [1/$((${#K_VALUES[@]} + 1))] Running BASELINE (no spec decode)..."
log_summary "Start time: $(date +"%Y-%m-%d %H:%M:%S")"

vllm bench throughput \
    --model "${TARGET_MODEL}" \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --trust-remote-code \
    --random-input-len ${INPUT_LEN} \
    --random-output-len ${OUTPUT_LEN} \
    --random-range-ratio 0.0 \
    --num-prompts ${NUM_PROMPTS} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --tensor-parallel-size ${TP_SIZE} \
    > "${BASELINE_LOG}" 2>&1

BASELINE_EXIT=$?
log_summary "End time:   $(date +"%Y-%m-%d %H:%M:%S")"
log_summary "Exit code:  ${BASELINE_EXIT}"
extract_metrics "BASELINE" "${BASELINE_LOG}"

# ---------- 2. Spec decoding with k=1..7 ----------
IDX=2
for K in "${K_VALUES[@]}"; do
    SPEC_LOG="${LOG_DIR}/spec_k${K}.log"
    log_summary ""
    log_summary ">>> [${IDX}/$((${#K_VALUES[@]} + 1))] Running SPEC DECODE k=${K}..."
    log_summary "Start time: $(date +"%Y-%m-%d %H:%M:%S")"

    vllm bench throughput \
        --model "${TARGET_MODEL}" \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code \
        --random-input-len ${INPUT_LEN} \
        --random-output-len ${OUTPUT_LEN} \
        --random-range-ratio 0.0 \
        --num-prompts ${NUM_PROMPTS} \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --tensor-parallel-size ${TP_SIZE} \
        --speculative-config "{
            \"method\": \"draft_model\",
            \"model\": \"${DRAFT_MODEL}\",
            \"num_speculative_tokens\": ${K},
            \"max_model_len\": ${MAX_MODEL_LEN}
        }" \
        > "${SPEC_LOG}" 2>&1

    SPEC_EXIT=$?
    log_summary "End time:   $(date +"%Y-%m-%d %H:%M:%S")"
    log_summary "Exit code:  ${SPEC_EXIT}"
    extract_metrics "SPEC k=${K}" "${SPEC_LOG}"

    IDX=$((IDX + 1))

    # 让显存和缓存释放一下，避免相邻实验互相影响
    sleep 5
done

# ---------- 最终汇总 ----------
log_summary ""
log_summary "============================================================"
log_summary "ALL DONE. Full logs saved under: ${LOG_DIR}"
log_summary "Summary log: ${SUMMARY_LOG}"
log_summary "============================================================"

# 再打印一份纯净的 Throughput 对比表到 stdout 和 summary
echo ""
echo "=============== Final Throughput Comparison ==============="
{
    echo ""
    echo "=============== Final Throughput Comparison ==============="
    printf "%-12s | %s\n" "Config" "Throughput line"
    printf -- "-------------|-------------------------------------------------\n"
    for f in "${BASELINE_LOG}" "${LOG_DIR}"/spec_k*.log; do
        tag=$(basename "$f" .log)
        tp=$(grep -E "Throughput:" "$f" | tail -n 1)
        printf "%-12s | %s\n" "${tag}" "${tp:-<no throughput line found>}"
    done
} | tee -a "${SUMMARY_LOG}"
