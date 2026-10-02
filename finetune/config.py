import os

# Configuration for the data. Index paths are env-overridable so an alternate corpus
# (e.g. the 50k-Irish build's isolated indices) can be trained without editing this file.
DATA_TRAIN_INDEX_PATH = os.environ.get(
    "DATA_TRAIN_INDEX_PATH", "../data/abcfiles_processed_v1_train.jsonl")   # V:1-only (melody) dataset
DATA_EVAL_INDEX_PATH = os.environ.get(
    "DATA_EVAL_INDEX_PATH", "../data/abcfiles_processed_v1_eval.jsonl")

# Configuration for the model 
PATCH_STREAM = True                                              # Stream training / inference
PATCH_SIZE = 16                                                  # Patch Size
PATCH_LENGTH = 1024                                              # Patch Length
CHAR_NUM_LAYERS = 6                                              # Number of layers in the decoder
PATCH_NUM_LAYERS = 20                                            # Number of layers in the encoder
HIDDEN_SIZE = 1280                                               # Hidden Size

# Backbone for each level of the hierarchy. See notagen_core.available_backbones()
# ("gpt2", "llama", ...). "gpt2" reproduces the original NotaGen and loads legacy
# checkpoints strict; any other backbone trains from scratch (different keys).
ENCODER_BACKBONE = "gpt2"                                        # patch-level encoder
DECODER_BACKBONE = "gpt2"                                        # char-level decoder

# Stage 2 toggle: when False (default) fine-tune the original gpt2 NotaGen from
# PRETRAINED_PATH. When True, start from the Stage-1 distilled larger patch
# encoder (+ projection) and the teacher's char decoder, then fine-tune
# end-to-end. The wider encoder feeds the 1280-d char decoder via an
# auto-inserted projection (NotaGenLMHeadModel.patch_proj).
USE_STAGE1_STUDENT       = False
STAGE1_STUDENT_PATH      = "/usr/xtmp/cy232/accompgen/distil/stage1_student"
STUDENT_ENCODER_BACKBONE = "llama"
STUDENT_PATCH_NUM_LAYERS = 24
STUDENT_HIDDEN_SIZE      = 1536

# PATCH_STREAM = True                                          # Stream training / inference
# PATCH_SIZE = 16                                                 # Patch Size
# PATCH_LENGTH = 2048                                             # Patch Length
# CHAR_NUM_LAYERS = 3                                             # Number of layers in the decoder
# PATCH_NUM_LAYERS = 12                                           # Number of layers in the encoder
# HIDDEN_SIZE = 768                                               # Hidden Size

# Configuration for the training
BATCH_SIZE = 1         
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 1e-5))   # env-overridable (LoRA runs use ~2e-4)

# LoRA fine-tuning (finetune/train-gen.py): when USE_LORA=1 the base NotaGen weights are
# frozen and rank-LORA_R adapters are trained on the GPT2 attention/MLP projections of
# BOTH the patch-level encoder and the char-level decoder. Checkpoints still store the
# full MERGED weights under 'model' (so every inference loader works unchanged) plus the
# adapter alone under 'lora' (for resuming).
USE_LORA = os.environ.get('USE_LORA', '0') == '1'
LORA_R = int(os.environ.get('LORA_R', 16))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', 32))
LORA_DROPOUT = float(os.environ.get('LORA_DROPOUT', 0.05))
LORA_TARGET_MODULES = ['c_attn', 'c_proj', 'c_fc']
NUM_EPOCHS_SYNTHETIC = int(os.environ.get('NUM_EPOCHS_SYNTHETIC', 2))        # Phase 1: epochs on synthetic data (multi-length set is ~7x bigger, so 2 epochs
                                                                              #  ≈ 14 passes over each source tunebody; was 10 on the length-4-only set)
NUM_EPOCHS_REAL = int(os.environ.get('NUM_EPOCHS_REAL', 5))                   # Phase 2: epochs on real (non-synthetic) data (7 length-variants per piece, so
                                                                              #  5 epochs ≈ 35 passes per tunebody; was 10 on the single-variant set)
                                                              # (env-overridable: larger corpora (e.g. 50k Irish) need fewer epochs to fit
                                                              #  walltime; eval bottomed at ~3 real epochs and rose after — see 1motif run)
# When True, skip the synthetic phase and resume straight into the real phase from the
# saved synthetic checkpoint at WEIGHTS_PATH (requires LOAD_FROM_CHECKPOINT=True). Used to
# resume after the real phase OOM'd without redoing the synthetic phase.
SKIP_SYNTHETIC_PHASE = os.environ.get('SKIP_SYNTHETIC_PHASE', '0') == '1'  # env-overridable for continuation runs
ACCUMULATION_STEPS = 1                                          # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training, 0 for full conaudio
MOTIF_LOSS_WEIGHT = 3.0                                         # (kept for weight loading compat; no longer used in loss)
MOTIF_ATTENTION_BIAS = float(os.environ.get('MOTIF_ATTENTION_BIAS', 4.0))  # Scalar added to QK attention logits for motif patch key positions before softmax
                                                                # (env-overridable so the bias-ablation runs can train with 0.0 without editing this file)
                                                                # (dose-response ladder: trained +2 -> 34%, +4 -> 44% motif containment optimum; +8 backfires/collapses)
LOAD_FROM_CHECKPOINT = os.environ.get('LOAD_FROM_CHECKPOINT', '0') == '1'  # Whether to load weights from a checkpoint (env-overridable)
WANDB_LOGGING = True                                            # Whether to log to wandb
WANDB_KEY = os.environ.get('WANDB_KEY', '')  # leave empty to use `wandb login` / WANDB_API_KEY

# PRETRAINED_PATH = "../weights/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"                # Path of pretrained weights
PRETRAINED_PATH = "/usr/xtmp/cy232/accompgen/pretrain/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"
EXP_TAG = os.environ.get('EXP_TAG', 'multilen4to10_v1_bias4_syn2real5')  # = multi-length expansion: real+synthetic data annotated at EVERY motif
                                                              # length 4..10 (7 real variants/piece, 3 crops/piece/length), trained 2 synthetic
                                                              # + 5 real epochs. Previous tag ('1motif_v1_bias4_cropsyn_irish20k_evalfix'):
                                                              # = irish20k rerun with the matched-eval fixes:
                                                              #   (1) synthetic phase evaluated on synthetic-only eval data (was full/mixed),
                                                              #   (2) eval applies the motif attention bias in any phase that trains with it,
                                                              #       so real-phase eval matches real-phase training AND inference (generate()).
                                                              # New _evalfix tag so it does NOT clobber the original irish20k weights/log (kept for
                                                              # before/after comparison). = cropsyn run + 20k Irishman tunes merged into the real V:1
                                                              # set (HF sander-wood/irishman; see scripts/run_preprocess_irishman.sh). bias4 (optimum)
                                                              # held fixed. Synthetic set also crops from the Irishman tunes (63k pieces).
NAME =  EXP_TAG + \
        (f"_lora{LORA_R}" if USE_LORA else "") + \
        "_p_size_" + str(PATCH_SIZE) + \
        "_p_length_" + str(PATCH_LENGTH) + \
        "_p_layers_" + str(PATCH_NUM_LAYERS) + \
        "_c_layers_" + str(CHAR_NUM_LAYERS) + \
        "_h_size_" + str(HIDDEN_SIZE) + \
        "_lr_" + str(LEARNING_RATE) + \
        "_batch_" + str(BATCH_SIZE)

BASE_WEIGHTS_PATH = "/usr/xtmp/cy232/accompgen/weights/"

WEIGHTS_PATH = BASE_WEIGHTS_PATH + "weights_notagen_" + NAME + ".pth"                  # Path to save weights
LOGS_PATH    = BASE_WEIGHTS_PATH + "logs_notagen_"    + NAME + ".txt"                     # Path to save logs
WANDB_NAME = NAME
