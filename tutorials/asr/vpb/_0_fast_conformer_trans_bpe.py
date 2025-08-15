import os
import gc
import argparse
import torch
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from nemo.utils import exp_manager
import nemo.collections.asr as nemo_asr
from pytorch_lightning.loggers import CSVLogger
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


def prepare_tokenizer(manifest_path, tokenizer_dir, vocab_size=128, tokenizer_type='spe', spe_type='unigram'):
    print("🔧 Preparing tokenizer...")

    document_dir = os.path.join(tokenizer_dir, 'text_corpus')
    document_path = os.path.join(document_dir, 'document.txt')
    os.makedirs(document_dir, exist_ok=True)

    if not os.path.exists(document_path):
        print(f"📄 Building text corpus from manifest: {manifest_path}")
        with open(manifest_path, 'r', encoding='utf-8') as f_in, open(document_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                item = eval(line)
                f_out.write(item['text'] + '\n')

    if tokenizer_type == 'spe':
        tokenizer_path = os.path.join(tokenizer_dir, f"tokenizer_spe_{spe_type}_v{vocab_size}")
        if os.path.exists(os.path.join(tokenizer_path, 'tokenizer.model')):
            print(f"⚠️ Tokenizer already exists at {tokenizer_path}, skipping creation.")
        else:
            print(f"🔠 Creating SentencePiece tokenizer at: {tokenizer_path}")
            os.makedirs(tokenizer_path, exist_ok=True)
            create_spt_model(
                data_file=document_path,
                vocab_size=vocab_size,
                sample_size=-1,
                do_lower_case=False,
                output_dir=tokenizer_path,
                tokenizer_type=spe_type,
                character_coverage=1.0,
                train_extremely_large_corpus=False,
                max_sentencepiece_length=-1,
                split_by_unicode_script=True,
                bos=False,
                eos=False,
                pad=False,
                control_symbols=None,
                user_defined_symbols=None,
                byte_fallback=False,
                split_digits=False,
                remove_extra_whitespaces=False,
            )
        tokenizer_type_cfg = 'bpe'
    else:
        raise NotImplementedError("Only SentencePiece tokenizer is supported in this refactor.")

    print(f"✅ Tokenizer ready at: {tokenizer_path} ({tokenizer_type_cfg})")
    return tokenizer_path, tokenizer_type_cfg

def configure_model(config_path, train_manifest, test_manifest, tokenizer_path, tokenizer_type_cfg, train_samples, num_gpus, epochs):
    print("🧩 Configuring model...")
    config = OmegaConf.load(config_path)

    config.model.sample_rate = 16000
    config.model.train_ds.batch_size = 32
    config.model.train_ds.max_duration = 17.125
    config.model.train_ds.manifest_filepath = train_manifest
    config.model.train_ds.bucketing_strategy = "fully_randomized"
    config.model.validation_ds.manifest_filepath = test_manifest
    config.model.test_ds.manifest_filepath = test_manifest

    config.model.tokenizer.dir = tokenizer_path
    config.model.tokenizer.type = tokenizer_type_cfg

    config.model.spec_augment.freq_masks = 0
    config.model.spec_augment.time_masks = 0

    config.model.joint.fuse_loss_wer = True
    config.model.joint.fused_batch_size = 16
    config.model.model_defaults.pred_hidden = 320
    config.model.model_defaults.joint_hidden = 320

    config.trainer.devices = num_gpus
    config.trainer.strategy = 'auto'
    config.trainer.precision = 16
    config.trainer.accumulate_grad_batches = 1

    actual_batch_size = config.model.train_ds.batch_size * num_gpus * config.trainer.accumulate_grad_batches
    base_lr = 1e-3
    config.model.optim.name = "adamw"
    config.model.optim.lr = base_lr * (actual_batch_size / 256)
    config.model.optim.sched.min_lr = config.model.optim.lr * 0.01
    config.model.optim.betas = [0.9, 0.999]
    config.model.optim.weight_decay = 0.0001

    warmup_steps = int(0.05 * epochs * train_samples / actual_batch_size)
    config.model.optim.sched.warmup_steps = warmup_steps
    config.model.log_prediction = False

    config.exp_manager.create_tensorboard_logger = False
    config.exp_manager.create_wandb_logger = False

    config.model.encoder.n_layers = 6
    config.model.encoder.d_model = 176
    config.model.encoder.n_heads = 1

    print(f"📐 Effective batch size: {actual_batch_size}")
    print(f"📈 Learning rate: {config.model.optim.lr:.6f}, Warmup steps: {warmup_steps}")
    return config

def build_trainer(log_dir, exp_name, epochs, accelerator='gpu'):
    print("🚀 Building Trainer...")
    logger = TensorBoardLogger(log_dir, name=exp_name, log_graph=True)
    trainer = Trainer(
        devices=-1,
        accelerator=accelerator,
        max_epochs=epochs,
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
    )
    return trainer

def setup_experiment(trainer):
    print("🗂️  Setting up experiment manager...")
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
    return exp_manager.exp_manager(trainer, OmegaConf.structured(exp_config))


def log_val_metrics_to_txt(log_dir, epoch, wer, loss):
    with open(f"{log_dir}/val_metrics.txt", "a") as f:
        f.write(f"Epoch {epoch:03d} | val_WER: {wer:.4f} | val_loss: {loss:.4f}\n")


def main():
    print("🧪 Starting NeMo ASR pipeline...")

    # ==== Config ====
    train_manifest = "datasets/vivos/train_manifest.json"
    test_manifest = "datasets/vivos/test_manifest.json"
    tokenizer_dir = "tokenizers"
    config_path = "tutorials/asr/configs/fast-conformer_transducer_bpe.yaml"
    exp_dir = "experiments"
    exp_name = "vpb_asr_fastconformer_transducer_bpe"
    num_gpus = 1
    epochs = 100

    # ==== Load and count samples ====
    with open(train_manifest, "r", encoding="utf-8") as f:
        train_samples = sum(1 for _ in f)
    print("📊 Number of training samples:", train_samples)

    # ==== Prepare tokenizer & config ====
    tokenizer_path, tokenizer_type_cfg = prepare_tokenizer(train_manifest, tokenizer_dir)
    config = configure_model(
        config_path, 
        train_manifest, test_manifest, 
        tokenizer_path, tokenizer_type_cfg, 
        train_samples, num_gpus, epochs)

    # ==== Logger CSV + Trainer ====
    # ==== Logger + Trainer ====

    trainer = build_trainer(
        log_dir=os.path.join(exp_dir, exp_name),
        exp_name=exp_name,
        epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )

    # 2. Gọi exp_manager để nó tạo thư mục và TensorBoard logger
    # Sau bước này, trainer.logger sẽ là TensorBoard logger
    # exp_manager.exp_manager(trainer, config.get("exp_manager"))

    # 3. TẠO VÀ THÊM CSV LOGGER
    print("📝 Adding CSV Logger...")
    # Lấy đường dẫn thư mục log mà exp_manager vừa tạo
    log_dir = trainer.log_dir 

    # # Tạo một CSVLogger và lưu vào cùng thư mục đó
    # # name="" và version="" để không tạo thêm thư mục con bên trong
    csv_logger = CSVLogger(save_dir=log_dir, name="", version="")

    # # Thêm csv_logger vào danh sách các logger của trainer
    # # Trainer giờ sẽ ghi log ra cả TensorBoard và file CSV
    trainer.loggers.append(csv_logger)

    # ==== Load Model ====
    print("🧠 Initializing model...")
    model = nemo_asr.models.EncDecRNNTBPEModel(cfg=config.model, trainer=trainer)

    # ==== Clean memory ====
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 100)
    print(model.summarize(max_depth=4))

    # ==== Training ====
    print("🏋️ Starting training...")
    trainer.fit(model)

    val_wer = trainer.callback_metrics["val_wer"].item()
    val_loss = trainer.callback_metrics["val_loss"].item() if "val_loss" in trainer.callback_metrics else 0.0
    log_val_metrics_to_txt(trainer.log_dir, trainer.current_epoch, val_wer, val_loss)


    # ==== Testing ====
    print("🔍 Running test...")
    trainer.test(model)


if __name__ == '__main__':
    main()

