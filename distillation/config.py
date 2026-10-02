import os

# ── NotaGen teacher (pre-trained, no motif conditioning) ─────────────────────
NOTAGEN_WEIGHTS_PATH = "/usr/xtmp/cy232/accompgen/pretrain/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"

# NotaGen architecture — must match the checkpoint
PATCH_STREAM             = True
PATCH_SIZE               = 16
PATCH_LENGTH             = 1024
CHAR_NUM_LAYERS          = 6
PATCH_NUM_LAYERS         = 20
HIDDEN_SIZE              = 1280
PATCH_SAMPLING_BATCH_SIZE = 0    # not used during inference; kept so CharLevelDecoder imports cleanly

# Teacher backbone per level. "gpt2" reproduces the original NotaGen teacher and
# loads its legacy checkpoint strict (see notagen_core.available_backbones()).
ENCODER_BACKBONE = "gpt2"
DECODER_BACKBONE = "gpt2"
MOTIF_ATTENTION_BIAS = 0.0       # teacher was pre-trained without motif conditioning

# ── Corpus generation ─────────────────────────────────────────────────────────
CORPUS_OUTPUT_PATH = "../data/distil_corpus.jsonl"
REAL_TRAIN_JSONL   = "../data/abcfiles_processed_train.jsonl"
REAL_EVAL_JSONL    = "../data/abcfiles_processed_eval.jsonl"

# Generate this many (prompt, completion) pairs in total
NUM_CORPUS_SAMPLES = 5000

# Temperatures used per prompt: first = "chosen" (low), last = "rejected" (high) for DPO
GEN_TEMPERATURES = [0.8, 1.2, 1.6]
GEN_TOP_K        = 9
GEN_TOP_P        = 0.9
GEN_TIMEOUT_SECS = 10 * 60     # abandon a single generation after 10 min

# ── Qwen student ──────────────────────────────────────────────────────────────
QWEN_MODEL_ID           = "Qwen/Qwen2.5-3B-Instruct"
QWEN_SFT_OUTPUT_PATH    = "/usr/xtmp/cy232/accompgen/distil/qwen_sft"
QWEN_MOTIF_OUTPUT_PATH  = "/usr/xtmp/cy232/accompgen/distil/qwen_motif"

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# ── Phase 1 — SFT on distillation corpus ─────────────────────────────────────
SFT_BATCH_SIZE        = 4
SFT_ACCUMULATION_STEPS = 4
SFT_LR                = 2e-4
SFT_EPOCHS            = 3
SFT_WARMUP_STEPS      = 100
SFT_MAX_SEQ_LEN       = 4096

# ── Phase 2 — Motif fine-tuning ───────────────────────────────────────────────
MOTIF_BATCH_SIZE           = 2
MOTIF_ACCUMULATION_STEPS   = 8
MOTIF_LR                   = 1e-4
MOTIF_EPOCHS_SYNTHETIC     = 20
MOTIF_EPOCHS_REAL          = 70
MOTIF_WARMUP_STEPS         = 500
MOTIF_MAX_SEQ_LEN          = 4096

# ── Teacher generation corpus (prompt across all valid combinations) ──────────
# Prompt the frozen teacher with every Period_Composer_Instrumentation combo and
# synthesize new ABC pieces as extra training data. The 112 combos live in
# gradio/prompts.txt; GEN_PER_COMBO × 112 sets the corpus size.
PROMPTS_FILE   = "../gradio/prompts.txt"
GEN_CORPUS_DIR = "/usr/xtmp/cy232/accompgen/distil/gen_corpus"
GEN_PER_COMBO  = 268             # × 112 combos ≈ 30k pieces (generation is the bottleneck — shard it)

# ── Stage 1 — patch-encoder feature distillation ─────────────────────────────
# Distill the frozen GPT-2 patch-level encoder's per-patch hidden states into a
# LARGER student encoder, then reuse the original (frozen) char-level decoder.
# No tokenizer mismatch and no KL needed: the target is a continuous feature
# vector, matched with MSE + cosine (FitNets-style hidden-state distillation).
# Intermediate training data (cached teacher features) lives under /usr/xtmp.
STAGE1_CACHE_DIR    = "/usr/xtmp/cy232/accompgen/distil/stage1_cache"     # cached (patches, teacher_feats)
STAGE1_OUTPUT_DIR   = "/usr/xtmp/cy232/accompgen/distil/stage1_student"   # trained student encoder + projection
STAGE1_USE_GEN_CORPUS = True     # also distill on the teacher-generated pieces in GEN_CORPUS_DIR

# Larger student patch encoder (see notagen_core.available_backbones()).
STUDENT_ENCODER_BACKBONE = "llama"
STUDENT_PATCH_NUM_LAYERS = 24
STUDENT_HIDDEN_SIZE      = 1536      # projected down to HIDDEN_SIZE (1280) for the frozen char decoder

# Precompute (caching) settings
STAGE1_PRECOMPUTE_BATCH  = 4
STAGE1_PASSES            = 1        # encoding passes per file; >1 adds random-transposition augmentation

# Training settings
STAGE1_BATCH_SIZE   = 2
STAGE1_ACCUM_STEPS  = 8
STAGE1_LR           = 2e-4
STAGE1_EPOCHS       = 10
STAGE1_WARMUP_STEPS = 200
STAGE1_COS_WEIGHT   = 1.0          # weight on the (1 - cosine) term alongside MSE
STAGE1_LAMBDA_LM    = 0.0          # weight on end-to-end char-LM loss through the frozen char decoder (0 = pure feature distill)

# ── Inference (demo) ──────────────────────────────────────────────────────────
INFER_TOP_K        = 50
INFER_TOP_P        = 0.9
INFER_TEMPERATURE  = 1.2
INFER_MAX_NEW_TOKENS = 4096
