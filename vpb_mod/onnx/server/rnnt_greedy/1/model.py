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

    def execute_v1(self, requests):
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



    def execute(self, requests):
        """
        Hợp nhất tất cả requests vào một batch lớn:
        - Pad AUDIO_SIGNAL theo T_max
        - Giữ AUDIO_LENGTH gốc cho từng sample
        - Chạy preprocessor/onnx/decoder 1 lần
        - Cắt kết quả trả lại từng request theo đúng kích thước batch của request đó
        """
        # 1) Thu thập dữ liệu từ mọi request
        per_req_batch = []     # số mẫu trong từng request
        all_wavs = []          # list[np.ndarray (Ti,)]
        all_lens = []          # list[int]
        for req in requests:
            x = pb_utils.get_input_tensor_by_name(req, "AUDIO_SIGNAL").as_numpy()  # (T,) hoặc (B,T)
            L = pb_utils.get_input_tensor_by_name(req, "AUDIO_LENGTH").as_numpy()  # (1,) hoặc (B,1) / int32
            # Chuẩn hoá shape
            if x.ndim == 1:
                x = x[None, :]
            if L.ndim == 1:
                L = L[:, None]
            B = x.shape[0]
            per_req_batch.append(B)
            # tách từng sample để quản lý độ dài riêng
            for i in range(B):
                wav_i = x[i]
                len_i = int(L[i, 0])
                # cắt an toàn nếu độ dài thừa
                wav_i = wav_i[:len_i]
                all_wavs.append(wav_i.astype(np.float32, copy=False))
                all_lens.append(len_i)

        # 2) Pad về cùng chiều (N, T_max)
        N = len(all_wavs)
        if N == 0:
            return [pb_utils.InferenceResponse(error=pb_utils.TritonError("Empty batch"))]
        T_max = max(all_lens)
        # Dùng Torch trực tiếp để tránh copy thêm lần nữa
        with torch.no_grad():
            sig = torch.zeros((N, T_max), dtype=torch.float32, device=self.device)
            for i, wav_i in enumerate(all_wavs):
                Ti = wav_i.shape[0]
                if Ti > 0:
                    sig[i, :Ti] = torch.from_numpy(wav_i).to(self.device)
            sig_len = torch.tensor(all_lens, dtype=torch.int64, device=self.device)

            # 3) Preprocess 1 lần
            mel, mel_len = self.asr_model.preprocessor(input_signal=sig, length=sig_len)

            # 4) ONNX greedy decode (encoder + joint) 1 lần
            hyps = self.decoding_onnx(audio_signal=mel, length=mel_len)

            # 5) NeMo decode -> texts (list[str] độ dài N)
            decoded = self.asr_model.decoding.decode_hypothesis(hyps)
            if isinstance(decoded, list):
                texts_all = [getattr(h, "text", str(h)) for h in decoded]
            else:
                texts_all = [getattr(decoded, "text", str(decoded))]

        # 6) Chia kết quả lại theo từng request & tạo response
        responses = []
        offset = 0
        for req, bsz in zip(requests, per_req_batch):
            chunk = texts_all[offset: offset + bsz]
            offset += bsz
            # TYPE_BYTES (khuyên dùng) -> encode UTF-8; nếu bạn dùng TYPE_STRING thì bỏ encode()
            out_np = np.array([t.encode("utf-8") for t in chunk], dtype=np.object_)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("TRANSCRIPT", out_np)]
            ))
        return responses
