import os
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.utils import exp_manager
from lightning.pytorch.loggers import TensorBoardLogger

import gc

# Preparing the dataset
TRAIN_MANIFEST = "datasets/vivos/train_manifest.json"
TEST_MANIFEST = "datasets/vivos/test_manifest.json"
EPOCHS = 300
num_gpus = 1

train_samples = 0
with open("datasets/vivos/train_manifest.json", "r", encoding="utf-8") as f:
    train_samples = sum(1 for _ in f)
print("Number of samples in train_manifest.json:", train_samples)

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
config = OmegaConf.load("configs/fast-conformer_transducer_bpe.yaml")
# dataset config
config.model.sample_rate = 16000
config.model.train_ds.batch_size = 32 # GPU MEM 16G, precision: 16
config.model.train_ds.max_duration = 17.125 # 17.125 seconds is the max duration in vivos dataset
config.model.train_ds.manifest_filepath = TRAIN_MANIFEST
config.model.train_ds.bucketing_strategy = "fully_randomized"
config.model.validation_ds.manifest_filepath = TEST_MANIFEST
config.model.test_ds.manifest_filepath = TEST_MANIFEST
# Tokenizer config
config.model.tokenizer.dir = TOKENIZER
config.model.tokenizer.type = TOKENIZER_TYPE_CFG




# augument
config.model.spec_augment.freq_masks = 0
config.model.spec_augment.time_masks = 0


# Enable fused batch step
config.model.joint.fuse_loss_wer = True
config.model.joint.fused_batch_size = 16

# Reduce hidden dimensions
config.model.model_defaults.pred_hidden = 320
config.model.model_defaults.joint_hidden = 320

# trainer config
config.trainer.devices = num_gpus  # Use all available GPUs
config.trainer.strategy = 'auto'  # Automatically select the best strategy based on available resources
# config.trainer.strategy="ddp"
config.trainer.precision = 16  # Use mixed precision training
config.trainer.accumulate_grad_batches = 1

# calculate learning_rate
# actual_batch_size = train_ds.batch_size × num_gpus × accumulate_grad_batches
# base_lr = 1e-3
# lr = base_lr x (actual_batch_size/256) 
actual_batch_size = config.model.train_ds.batch_size * num_gpus * config.trainer.accumulate_grad_batches
lr = 1e-3*(actual_batch_size/256)
print("learning_rate: ", lr)
config.model.optim.name="adamw"
config.model.optim.lr=lr # 0.01
config.model.optim.sched.min_lr=lr*0.01 
config.model.optim.betas=[0.9,0.999]
config.model.optim.weight_decay=0.0001

# calculate warnup
# total_steps = (dataset_size / actual_batch_size) × max_epochs
# warmup_steps = 0.05 × total_steps
warnup_steps = int(0.05*EPOCHS*train_samples/actual_batch_size)
print("warnup_steps: ", warnup_steps)
config.model.log_prediction = False
config.model.optim.sched.warmup_steps = warnup_steps
config.exp_manager.create_wandb_logger=False
config.exp_manager.create_tensorboard_logger=False

# encoder config
config.model.encoder.n_layers = 6
config.model.encoder.d_model = 176
config.model.encoder.n_heads = 1




# Initialize trainer
if torch.cuda.is_available():
    accelerator = 'gpu'
else:
    accelerator = 'gpu'

exp_dir = "experiments/"
exp_name = "vpb_asr_fastconformer_transducer_bpe"

tb_logger = TensorBoardLogger(exp_dir, name=exp_name, log_graph=True)


trainer = Trainer(
    devices=-1,
    accelerator=accelerator,
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    logger=tb_logger,
    log_every_n_steps=10,
    check_val_every_n_epoch=5,
    # precision=16,
)

# Initialize the model
model = nemo_asr.models.EncDecRNNTBPEModel(cfg=config.model, trainer=trainer)
# model.summarize()
# print(model)

# Load pre-trained weights if available
ckpt_dir = "eexperiments/vpb_asr_fastconformer_transducer_bpe/2025-07-21_09-23-26/checkpoints/"
# model.load_from_checkpoint('path_to_pretrained_model.ckpt')
# pretrained_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("stt_en_fastconformer_transducer_large", map_location='cpu')
# pretrained_model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint(os.path.join(ckpt_dir,vpb_asr_fastconformer_transducer_bpe--val_wer=0.3318-epoch=100-last.ckptt"), map_location='cpu')
# model.load_state_dict(pretrained_model.state_dict(), strict=False)
# model.encoder.load_state_dict(pretrained_model.encoder.state_dict(), strict=True)

# Configure experiment manager
os.environ.pop('NEMO_EXPM_VERSION', None)

exp_config = exp_manager.ExpManagerConfig(
    exp_dir=None,
    name=None,
    create_tensorboard_logger=False,
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
# trainer.fit(model)

# output = model.transcribe('datasets/vivos/test/augumented_8k_waves/VIVOSDEV01/VIVOSDEV01_R002.wav')
# print("Transcribe text: ", output)
print("="*100)
print(model.summarize(max_depth=6))
