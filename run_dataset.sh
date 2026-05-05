#!/bin/bash
set -e

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ===================================
# ONLY THESE ARE VARIED
# ===================================
SEEDS=(42)
USE_GT_LIST=(0 1)
DISABLE_CRF_LIST=(True False)

# ===================================
# DATASETS
# ===================================
DATASETS=("ACL" "NeurIPS" )

# ===================================
# GRID (UNCHANGED)
# ===================================
TOPK_INTRA_LIST=(4)
TOPM_TEMPORAL_LIST=(4)
HEADS_LIST=(8)
LAYERS_LIST=(4)
GATE_TAU_LIST=(0.2)
LR_LIST=(0.02)

# ===================================
# FIXED HYPERPARAMETERS (UNCHANGED)
# ===================================
ENCODER="all-mpnet-base-v2"
K=50
K_MAX=50
LAMBDA_GATE=0.01
EPOCHS=1000

# ===================================
# RESULTS ROOT
# ===================================
RESULTS_ROOT="Results"
mkdir -p "$RESULTS_ROOT"

echo "========================================"
echo "GRID SEARCH STARTED"
echo "========================================"

# ===================================
# SEED LOOP
# ===================================
for SEED in "${SEEDS[@]}"; do

# ===================================
# GT LOOP
# ===================================
for USE_GT in "${USE_GT_LIST[@]}"; do

# ===================================
# CRF LOOP
# ===================================
for DISABLE_CRF in "${DISABLE_CRF_LIST[@]}"; do

echo
echo "########################################"
echo "SEED: $SEED | USE_GT: $USE_GT | DISABLE_CRF: $DISABLE_CRF"
echo "########################################"

# ===================================
# DATASET LOOP
# ===================================
for DS in "${DATASETS[@]}"; do

echo
echo "########################################"
echo "DATASET: $DS"
echo "########################################"

# -----------------------------------
# Paths
# -----------------------------------
DATA_DIR="datasets/${DS}_jsonl"
VOCAB_FILE="datasets/${DS}/vocab.txt"
WORD_EMB_NPZ="datasets/${DS}/word_embeddings_encoder.npz"
TRAIN_TEXTS="datasets/${DS}/train_texts.txt"
TRAIN_TIMES="datasets/${DS}/train_times.txt"

# -----------------------------------
# Output directories
# Folder structure:
# Results/seed_<seed>/gt_<use_gt>_crf_<disable_crf>/<dataset>/
# -----------------------------------
MASTER_OUTROOT="${RESULTS_ROOT}/seed_${SEED}/gt_${USE_GT}_crf_${DISABLE_CRF}/${DS}"
mkdir -p "$MASTER_OUTROOT"

MASTER_RESULTS_TSV="$MASTER_OUTROOT/results_${DS}.tsv"
MASTER_SUMMARY_CSV="$MASTER_OUTROOT/summary_${DS}.csv"

echo -e "run_id\tseed\tuse_gt\tdisable_crf\ttopk\ttopm\thead\tlayers\tgate_tau\tlr\tcoherence\tdiversity" \
> "$MASTER_RESULTS_TSV"

RUN_ID=0

# ===================================
# GRID LOOP
# ===================================
for TOPK_INTRA in "${TOPK_INTRA_LIST[@]}"; do
for TOPM_TEMPORAL in "${TOPM_TEMPORAL_LIST[@]}"; do
for HEADS in "${HEADS_LIST[@]}"; do
for LAYERS in "${LAYERS_LIST[@]}"; do
for GATE_TAU in "${GATE_TAU_LIST[@]}"; do
for LR in "${LR_LIST[@]}"; do

RUN_ID=$((RUN_ID+1))

OUTROOT="$MASTER_OUTROOT/run_${RUN_ID}"
ARTIFACTS_DIR="$OUTROOT/artifacts"
CACHE_DIR="$OUTROOT/cache"
CACHE_TEST_DIR="$OUTROOT/cache_test"
EVAL_OUT_DIR="$OUTROOT/eval_out"

mkdir -p "$ARTIFACTS_DIR" "$CACHE_DIR" "$CACHE_TEST_DIR" "$EVAL_OUT_DIR"

echo
echo "==============================="
echo "Run $RUN_ID"
echo "Dataset: $DS"
echo "seed=$SEED use_gt=$USE_GT disable_crf=$DISABLE_CRF"
echo "topk=$TOPK_INTRA topm=$TOPM_TEMPORAL heads=$HEADS layers=$LAYERS gate_tau=$GATE_TAU lr=$LR"
echo "==============================="

# ===================================
# TRAIN
# ===================================
python -m scripts.run_train_gt \
--seed "$SEED" \
--use_gt "$USE_GT" \
--disable_crf "$DISABLE_CRF" \
--data_dir "$DATA_DIR" \
--cache_dir "$CACHE_DIR" \
--cache_test_dir "$CACHE_TEST_DIR" \
--artifacts_dir "$ARTIFACTS_DIR" \
--K "$K" \
--K_max "$K_MAX" \
--lambda_gate "$LAMBDA_GATE" \
--gate_tau "$GATE_TAU" \
--topk_intra "$TOPK_INTRA" \
--topm_temporal "$TOPM_TEMPORAL" \
--heads "$HEADS" \
--layers "$LAYERS" \
--lr "$LR" \
--epochs "$EPOCHS" \
--vocab_file "$VOCAB_FILE" \
--word_emb_npz "$WORD_EMB_NPZ" \
2>&1 | tee "$OUTROOT/train.log"

BEST_MODEL="$ARTIFACTS_DIR/best_model.pt"

# ===================================
# EVALUATION
# ===================================
if [ ! -f "$BEST_MODEL" ]; then

COHERENCE="NA"
DIVERSITY="NA"

else

python -m scripts.ANTM_style \
--model_ckpt "$BEST_MODEL" \
--vocab_file "$VOCAB_FILE" \
--word_emb_npz "$WORD_EMB_NPZ" \
--train_texts "$TRAIN_TEXTS" \
--train_times "$TRAIN_TIMES" \
--topn 15 \
--coherence c_v \
--save_dir "$EVAL_OUT_DIR" \
> "$OUTROOT/eval.log"

COHERENCE=$(grep -oP "Dynamic topic coherence \(c_v\) = \K[0-9]+\.[0-9]+" "$OUTROOT/eval.log" | tail -1)
DIVERSITY=$(grep -oP "Dynamic topic diversity \(refined\) = \K[0-9]+\.[0-9]+" "$OUTROOT/eval.log" | tail -1)

COHERENCE=${COHERENCE:-"NA"}
DIVERSITY=${DIVERSITY:-"NA"}

fi

echo -e "${RUN_ID}\t${SEED}\t${USE_GT}\t${DISABLE_CRF}\t${TOPK_INTRA}\t${TOPM_TEMPORAL}\t${HEADS}\t${LAYERS}\t${GATE_TAU}\t${LR}\t${COHERENCE}\t${DIVERSITY}" \
>> "$MASTER_RESULTS_TSV"

done
done
done
done
done
done

# ===================================
# SUMMARY
# ===================================
echo "run_id,seed,use_gt,disable_crf,coherence,diversity,combined_score" > "$MASTER_SUMMARY_CSV"

tail -n +2 "$MASTER_RESULTS_TSV" | while IFS=$'\t' read -r id seed use_gt disable_crf k m h l gt lr c d; do

if [[ "$c" =~ ^[0-9.]+$ ]] && [[ "$d" =~ ^[0-9.]+$ ]]; then
combined=$(echo "$c * $d" | bc -l)
printf "%s,%s,%s,%s,%.4f,%.4f,%.4f\n" "$id" "$seed" "$use_gt" "$disable_crf" "$c" "$d" "$combined" >> "$MASTER_SUMMARY_CSV"
else
printf "%s,%s,%s,%s,%s,%s,%s\n" "$id" "$seed" "$use_gt" "$disable_crf" "$c" "$d" "NA" >> "$MASTER_SUMMARY_CSV"
fi

done

echo
echo "===== DATASET $DS FINISHED ====="

column -t -s, "$MASTER_SUMMARY_CSV" 2>/dev/null || cat "$MASTER_SUMMARY_CSV"

echo "Output directory: $MASTER_OUTROOT"

done
done
done
done

echo
echo "========================================"
echo "ALL DATASETS FINISHED"
echo "========================================"
