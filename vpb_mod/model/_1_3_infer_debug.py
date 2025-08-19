from pathlib import Path
from omegaconf import OmegaConf
import torch
from lightning.pytorch import Trainer
import nemo.collections.asr as nemo_asr

nemo = Path("../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer.nemo")
dev = Path("~/work/public_datasets/vi_small/nemo_manifests/lsvsc/lsvsc_dev.jsonl").expanduser()

model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(nemo), map_location="cuda")
model.eval()

# test_ds = DEV (nếu so với val_wer ~0.137); nếu so test_wer thì đặt manifest sang lsvsc_test.jsonl
model.cfg.test_ds.manifest_filepath = str(dev)
model.cfg.test_ds.batch_size = 16     # khớp với training
model.cfg.test_ds.num_workers = 8
model.cfg.test_ds.return_transcripts = False
model.setup_test_data(model.cfg.test_ds)

# ép decoding = greedy_batch như validation
model.change_decoding_strategy(decoding_cfg=OmegaConf.create({"strategy":"greedy_batch"}))

trainer = Trainer(accelerator="gpu", devices=1, precision="16")
print(trainer.test(model))

