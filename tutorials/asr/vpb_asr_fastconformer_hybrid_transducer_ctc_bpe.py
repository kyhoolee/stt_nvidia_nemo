import os
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.utils import exp_manager
import gc

# Preparing the dataset
TRAIN_MANIFEST = "datasets/vivos/train_manifest.json"
TEST_MANIFEST = "datasets/vivos/test_manifest.json"

# Preparing the tokenizer
VOCAB_SIZE = 128  # can be any value above 29
TOKENIZER_TYPE = "spe"  # can be wpe or spe
SPE_TYPE = "unigram"  # can be bpe or unigram

# Clean and create tokenizer directory
os.system('rm -r tokenizers/')
if not os.path.exists("tokenizers"):
    os.makedirs("tokenizers")

# Process ASR text tokenizer
os.system(f'python scripts/process_asr_text_tokenizer.py \
   --manifest={TRAIN_MANIFEST} \
   --data_root="tokenizers" \
   --tokenizer={TOKENIZER_TYPE} \
   --spe_type={SPE_TYPE} \
   --no_lower_case \
   --log \
   --vocab_size={VOCAB_SIZE}')

# Set tokenizer path based on type
if TOKENIZER_TYPE == 'spe':
    TOKENIZER = os.path.join("tokenizers", f"tokenizer_spe_{SPE_TYPE}_v{VOCAB_SIZE}")
    TOKENIZER_TYPE_CFG = "bpe"
else:
    TOKENIZER = os.path.join("tokenizers", f"tokenizer_wpe_v{VOCAB_SIZE}")
    TOKENIZER_TYPE_CFG = "wpe"

# Load model config
# config = OmegaConf.load("configs/contextnet_rnnt.yaml")
config = OmegaConf.load("configs/fastconformer_hybrid_transducer_ctc_bpe.yaml")
# dataset config
config.model.sample_rate = 16000
config.model.train_ds.batch_size = 32 # GPU MEM 16G, precision: 16
config.model.train_ds.max_duration = 17.125 # 17.125 seconds is the max duration in vivos dataset
config.model.train_ds.manifest_filepath = TRAIN_MANIFEST
config.model.validation_ds.manifest_filepath = TEST_MANIFEST
config.model.test_ds.manifest_filepath = TEST_MANIFEST
# Tokenizer config
config.model.tokenizer.dir = TOKENIZER
config.model.tokenizer.type = TOKENIZER_TYPE_CFG

# Remove logging of samples and warmup since dataset is small
config.model.log_prediction = False
config.model.optim.sched.warmup_steps = None

config.model.spec_augment.freq_masks = 0
config.model.spec_augment.time_masks = 0


# Enable fused batch step
config.model.joint.fuse_loss_wer = True
config.model.joint.fused_batch_size = 16

# Reduce hidden dimensions
config.model.model_defaults.pred_hidden = 320
config.model.model_defaults.joint_hidden = 320

config.trainer.strategy="ddp"
    
config.model.optim.name="adamw"
config.model.optim.lr=0.01
config.model.optim.betas=[0.9,0.999]
config.model.optim.weight_decay=0.0001
config.model.optim.sched.warmup_steps=2000
config.exp_manager.create_wandb_logger=True

# encoder config
config.model.encoder.n_layers = 6
config.model.encoder.d_model = 176
config.model.encoder.n_heads = 1


# contextnet config
# config.model.encoder.jasper = config.model.encoder.jasper[:5]
# config.model.encoder.jasper[-1].filters = '${model.model_defaults.enc_hidden}'
# config.model.model_defaults.filters = 128

# trainer config
config.trainer.devices = -1  # Use all available GPUs
config.trainer.strategy = 'auto'  # Automatically select the best strategy based on available resources
config.trainer.precision = 16  # Use mixed precision training

# Initialize trainer
if torch.cuda.is_available():
    accelerator = 'gpu'
else:
    accelerator = 'gpu'

EPOCHS = 200

trainer = Trainer(
    devices=1,
    accelerator=accelerator,
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    logger=False,
    log_every_n_steps=10,
    check_val_every_n_epoch=5
)

# Initialize the model
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel(cfg=config.model, trainer=trainer)
# model.summarize()
# print(model)

# Load pre-trained weights if available
# ckpt_dir = "experiments/vpb_asr_fastconformer_transducer_bpe/2025-07-19_13-11-08/checkpoints/"
# model.load_from_checkpoint('path_to_pretrained_model.ckpt')
# pretrained_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained("stt_en_fastconformer_hybrid_large_pc", map_location='cpu')
# pretrained_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(os.path.join(ckpt_dir, "vpb_asr_fastconformer_transducer_bpe--val_wer=0.3982-epoch=10-last.ckpt"), map_location='cpu')
# model.load_state_dict(pretrained_model.state_dict(), strict=True)
# model.encoder.load_state_dict(pretrained_model.encoder.state_dict(), strict=True)

# Configure experiment manager
os.environ.pop('NEMO_EXPM_VERSION', None)

exp_config = exp_manager.ExpManagerConfig(
    exp_dir='experiments/',
    name="vpb_asr_fastconformer_transducer_bpe",
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
    ),
)

exp_config = OmegaConf.structured(exp_config)
logdir = exp_manager.exp_manager(trainer, exp_config)

# Clean up resources before training
gc.collect()
if accelerator == 'gpu':
    torch.cuda.empty_cache()

# Start training
trainer.fit(model)

output = model.transcribe('datasets/vivos/test/augumented_8k_waves/VIVOSDEV01/VIVOSDEV01_R002.wav')
print("Transcribe text: ", output)

