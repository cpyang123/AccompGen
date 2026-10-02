import os

# Configurations for inference
# Full MotiGen checkpoint (bias4 + curriculum, 4 syn + 4 real epochs on the
# full-Irishman corpus) -- the model the paper's demo serves.
INFERENCE_WEIGHTS_PATH = os.environ.get(
    'INFERENCE_WEIGHTS_PATH',
    '/usr/xtmp/cy232/accompgen/weights/weights_notagen_1motif_v1_bias4_cropsyn_irishfull_4x4'
    '_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth')
TOP_K = 9                                                       # Top k for sampling
TOP_P = 0.9                                                      # Top p for sampling
TEMPERATURE = 1.2                                                 # Temperature for sampling

# Configurations for model
PATCH_STREAM = True                                             # Stream training / inference
PATCH_SIZE = 16                                                # Patch Size
PATCH_LENGTH = 1024                                             # Patch Length
CHAR_NUM_LAYERS = 6                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 20                                           # Number of layers in the encoder
HIDDEN_SIZE = 1280                                               # Hidden Size

# Backbone per level. "gpt2" reproduces the original NotaGen and loads legacy
# checkpoints strict; see notagen_core.available_backbones() for alternatives.
ENCODER_BACKBONE = "gpt2"
DECODER_BACKBONE = "gpt2"
MOTIF_ATTENTION_BIAS = 4.0                                       # matched to the bias4-trained checkpoint (0 disables)
PATCH_SAMPLING_BATCH_SIZE = 0