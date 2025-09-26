# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import triton_python_backend_utils as pb_utils

# Import đúng lớp RNNT BPE để registry sẵn sàng
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import ONNXGreedyBatchedRNNTInfer


class TritonPythonModel:
    """Phiên bản v2: ưu tiên chạy CPU, giảm copy thừa và chuẩn bị sẵn cho GPU khi rảnh."""

    def initialize(self, args):
        # --- resolve model_dir an toàn ---
        model_name = args["model_name"]
        model_version = str(args["model_version"])
        repo = args["model_repository"]
        if os.path.basename(repo.rstrip("/")) == model_name:
            model_dir = os.path.join(repo, model_version)
        else:
            model_dir = os.path.join(repo, model_name, model_version)

        # --- đọc parameters từ config.pbtxt ---
        model_config = json.loads(args["model_config"])
        params = {k: v.get("string_value", "") for k, v in model_config.get("parameters", {}).items()}

        nemo_model_rel = params.get("nemo_model_path", "rnnt_asr.nemo")
        encoder_rel = params.get("encoder_path", "encoder-vpb_fastconformer.onnx")
        decoder_rel = params.get("decoder_path", "decoder_joint-vpb_fastconformer.onnx")

        self.nemo_path = os.path.join(model_dir, nemo_model_rel)
        self.encoder_path = os.path.join(model_dir, encoder_rel)
        self.decoder_path = os.path.join(model_dir, decoder_rel)

        # CPU/GPU tùy bạn (đảm bảo image có CUDA nếu dùng GPU)
        self.device = torch.device(os.environ.get("DEVICE", "cpu").lower())

        # --- Greedy RNNT ONNX (encoder + joint) ---
        max_symbols = int(os.environ.get("MAX_SYMBOLS_PER_STEP", "5"))
        self.decoding_onnx = ONNXGreedyBatchedRNNTInfer(
            self.encoder_path, self.decoder_path, max_symbols
        )

        # --- Load NeMo .nemo để dùng preprocessor + decoding/tokenizer ---
        self.asr_model = EncDecRNNTBPEModel.restore_from(self.nemo_path, map_location="cpu")
        self.asr_model.freeze()
        self.asr_model.eval()

        if hasattr(self.asr_model, "to"):
            self.asr_model.to(self.device)

        # Tắt augment / chuẩn hóa preproc cho inference
        try:
            if hasattr(self.asr_model, "spec_augmentation") and self.asr_model.spec_augmentation is not None:
                self.asr_model.spec_augmentation = None
            if hasattr(self.asr_model, "preprocessor"):
                preproc = self.asr_model.preprocessor
                if hasattr(preproc, "to"):
                    preproc.to(self.device)
                if hasattr(preproc, "dither"):
                    preproc.dither = 0.0
                if hasattr(preproc, "pad_to"):
                    preproc.pad_to = 0
        except Exception:
            pass

        # Cấu hình decoding nhẹ hơn (không timestamps/alignments)
        decoding_cfg = getattr(self.asr_model.decoding, "cfg", None)
        if decoding_cfg is not None:
            if hasattr(decoding_cfg, "compute_timestamps"):
                decoding_cfg.compute_timestamps = False
            if hasattr(decoding_cfg, "preserve_alignments"):
                decoding_cfg.preserve_alignments = False

        # Tránh logging metric không cần thiết trong Triton
        if hasattr(self.asr_model, "wer"):
            self.asr_model.wer = None

        # Bật cudnn benchmark khi có GPU rảnh
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Cache device flags cho câu lệnh khác
        self._needs_host_transfer = getattr(self.decoding_onnx, "device", "cpu") != "cuda"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _decode_texts(self, hyps):
        decoded = self.asr_model.decoding.decode_hypothesis(hyps)
        if isinstance(decoded, list):
            return [getattr(h, "text", str(h)) for h in decoded]
        return [getattr(decoded, "text", str(decoded))]

    def _batch_requests(self, requests):
        per_req_batch = []
        wav_tensors = []
        sample_lens = []

        for req in requests:
            sig_np = pb_utils.get_input_tensor_by_name(req, "AUDIO_SIGNAL").as_numpy()
            len_np = pb_utils.get_input_tensor_by_name(req, "AUDIO_LENGTH").as_numpy()

            if sig_np.ndim == 1:
                sig_np = sig_np[None, :]
            if len_np.ndim == 1:
                len_np = len_np[:, None]

            batch = sig_np.shape[0]
            per_req_batch.append(batch)

            for i in range(batch):
                effective_len = int(len_np[i, 0])
                wav = sig_np[i, :effective_len].astype(np.float32, copy=False)
                wav_tensor = torch.from_numpy(wav)
                wav_tensors.append(wav_tensor)
                sample_lens.append(effective_len)

        if not wav_tensors:
            raise ValueError("Empty batch")

        padded = pad_sequence(wav_tensors, batch_first=True, padding_value=0.0)
        lengths = torch.tensor(sample_lens, dtype=torch.int64)

        if self.device.type == "cuda":
            padded = padded.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)

        return padded, lengths, per_req_batch

    def _run_model(self, signals, signal_lens):
        with torch.inference_mode():
            mel, mel_len = self.asr_model.preprocessor(input_signal=signals, length=signal_lens)

            if mel.device.type == "cuda" and self._needs_host_transfer:
                mel_for_onnx = mel.detach().to("cpu")
            else:
                mel_for_onnx = mel

            mel_len_cpu = mel_len.detach().to("cpu")
            hyps = self.decoding_onnx(audio_signal=mel_for_onnx, length=mel_len_cpu)
            return self._decode_texts(hyps)

    def _build_responses(self, requests, per_req_batch, transcripts):
        responses = []
        offset = 0
        for req, batch in zip(requests, per_req_batch):
            chunk = transcripts[offset: offset + batch]
            offset += batch
            out_tensor = pb_utils.Tensor(
                "TRANSCRIPT", np.array([t.encode("utf-8") for t in chunk], dtype=np.object_)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    # ------------------------------------------------------------------
    # Triton entrypoints
    # ------------------------------------------------------------------
    def execute_v1(self, requests):
        try:
            signals, signal_lens, per_req_batch = self._batch_requests(requests)
            transcripts = self._run_model(signals, signal_lens)
            return self._build_responses(requests, per_req_batch, transcripts)
        except Exception as exc:  # pragma: no cover - Triton sẽ log lỗi
            err = pb_utils.TritonError(str(exc))
            return [pb_utils.InferenceResponse(error=err)]

    def execute(self, requests):
        """Bản mới: gom batch một lần cho toàn bộ request list."""
        try:
            signals, signal_lens, per_req_batch = self._batch_requests(requests)
            transcripts = self._run_model(signals, signal_lens)
            return self._build_responses(requests, per_req_batch, transcripts)
        except Exception as exc:  # pragma: no cover - Triton sẽ log lỗi
            err = pb_utils.TritonError(str(exc))
            return [pb_utils.InferenceResponse(error=err)]
