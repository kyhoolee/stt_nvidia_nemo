# -*- coding: utf-8 -*-
import os, json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

# Import đúng lớp RNNT BPE để registry sẵn sàng
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import ONNXGreedyBatchedRNNTInfer


class TritonPythonModel:
    def initialize(self, args):
        # --- resolve model_dir an toàn ---
        model_name    = args["model_name"]
        model_version = str(args["model_version"])
        repo          = args["model_repository"]
        if os.path.basename(repo.rstrip("/")) == model_name:
            model_dir = os.path.join(repo, model_version)
        else:
            model_dir = os.path.join(repo, model_name, model_version)

        # --- đọc parameters từ config.pbtxt ---
        model_config = json.loads(args["model_config"])
        params = {k: v.get("string_value","") for k, v in model_config.get("parameters", {}).items()}

        nemo_model_rel = params.get("nemo_model_path", "rnnt_asr.nemo")
        encoder_rel    = params.get("encoder_path", "encoder-vpb_fastconformer.onnx")
        decoder_rel    = params.get("decoder_path", "decoder_joint-vpb_fastconformer.onnx")

        self.nemo_path    = os.path.join(model_dir, nemo_model_rel)
        self.encoder_path = os.path.join(model_dir, encoder_rel)
        self.decoder_path = os.path.join(model_dir, decoder_rel)

        # CPU/GPU tùy bạn (đảm bảo image có CUDA nếu dùng GPU)
        self.device = torch.device(os.environ.get("DEVICE", "cpu"))

        # --- Greedy RNNT ONNX (encoder + joint) ---
        max_symbols = int(os.environ.get("MAX_SYMBOLS_PER_STEP", "5"))
        self.decoding_onnx = ONNXGreedyBatchedRNNTInfer(
            self.encoder_path, self.decoder_path, max_symbols
        )

        # --- Load NeMo .nemo để dùng preprocessor + decoding/tokenizer ---
        self.asr_model = EncDecRNNTBPEModel.restore_from(self.nemo_path, map_location=self.device)
        self.asr_model.freeze()

        # Tắt augment / chuẩn hóa preproc cho inference
        try:
            if hasattr(self.asr_model, "spec_augmentation") and self.asr_model.spec_augmentation is not None:
                self.asr_model.spec_augmentation = None
            if hasattr(self.asr_model, "preprocessor"):
                if hasattr(self.asr_model.preprocessor, "dither"):
                    self.asr_model.preprocessor.dither = 0.0
                if hasattr(self.asr_model.preprocessor, "pad_to"):
                    self.asr_model.preprocessor.pad_to = 0
        except Exception:
            pass

        # Đảm bảo decoding object sẵn sàng (sẽ dùng để decode hypotheses trả về từ ONNXGreedy)
        # Không cần change_decoding_strategy vì ta decode từ hyps đã có
        if hasattr(self.asr_model, "wer"):
            self.asr_model.wer = None  # tránh side effects logging trong env Triton

    def execute(self, requests):
        responses = []
        for req in requests:
            x = pb_utils.get_input_tensor_by_name(req, "AUDIO_SIGNAL").as_numpy()  # float32 mono PCM
            L = pb_utils.get_input_tensor_by_name(req, "AUDIO_LENGTH").as_numpy()  # int64 số mẫu

            # Chuẩn batch
            if x.ndim == 1:
                x = x[None, :]
            if L.ndim == 1:
                L = L[:, None]

            with torch.no_grad():
                sig     = torch.from_numpy(x).float().to(self.device)
                sig_len = torch.from_numpy(L.astype(np.int64)).to(self.device).squeeze(-1)

                # 1) Preprocess -> mel & length
                mel, mel_len = self.asr_model.preprocessor(input_signal=sig, length=sig_len)

                # 2) ONNX greedy decode (chạy encoder & joint onnx)
                hyps = self.decoding_onnx(audio_signal=mel, length=mel_len)

                # 3) NeMo decode hypotheses -> text
                decoded = self.asr_model.decoding.decode_hypothesis(hyps)
                if isinstance(decoded, list):
                    texts = [getattr(h, "text", str(h)) for h in decoded]
                else:
                    texts = [getattr(decoded, "text", str(decoded))]

            # Triton Python backend kỳ vọng BYTES -> encode UTF-8
            out = np.array([t.encode("utf-8") for t in texts], dtype=np.object_)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("TRANSCRIPT", out)]
            ))
        return responses
