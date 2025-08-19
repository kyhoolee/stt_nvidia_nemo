from pathlib import Path
from omegaconf import OmegaConf
import torch
from lightning.pytorch import Trainer
import nemo.collections.asr as nemo_asr
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

ckpt = Path("../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer--val_wer=0.1370-epoch=99.ckpt")
tok_dir = Path("/home/kylh/work/nemo_work/_1_small_vi_ds/tokenizers/lsvsc/tokenizer_spe_unigram_v256")
dev = Path("~/work/public_datasets/vi_small/nemo_manifests/lsvsc/lsvsc_dev.jsonl").expanduser()

# 1) Load ckpt
model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint(
    checkpoint_path=str(ckpt), map_location="cuda", strict=False
).eval()

# 2) Gắn lại tokenizer (quan trọng!)
model.cfg.tokenizer.type = "bpe"
model.cfg.tokenizer.dir = str(tok_dir)
model._setup_tokenizer(model.cfg.tokenizer)   # rebuild tokenizer from SPM model

# 3) Ép decoding giống lúc val (greedy_batch)
model.change_decoding_strategy(decoding_cfg=OmegaConf.create({"strategy": "greedy_batch"}))

# 4) Cấu hình test_ds y hệt lúc train
model.cfg.test_ds.manifest_filepath = str(dev)   # hoặc lsvsc_test.jsonl nếu so test_wer
model.cfg.test_ds.batch_size = 16
model.cfg.test_ds.num_workers = 8
model.cfg.test_ds.return_transcripts = False
model.setup_test_data(model.cfg.test_ds)

# 5) Run test
trainer = Trainer(accelerator="gpu", devices=1, precision="16")
print(trainer.test(model))
