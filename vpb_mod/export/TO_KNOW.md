tuyệt — mình tóm tắt “chân dung I/O” của model của bạn (từ log probe) thật gọn, đúng số liệu:

# Model-level

* Tokenizer: SentencePiece, vocab = **1024**
* `blank_id` = **1024**
* Prediction LSTM: **num\_layers = 1**, **hidden\_size = 640**
* Encoder dim (d\_model): **512**

# Preprocessor (AudioToMelSpectrogramPreprocessor)

* **forward signature:** `(input_signal, length)`
* **input:**

  * `input_signal`: `float32` `[B, T]` → ví dụ `(1, 24000)` cho 1.5s @16k
  * `length`: `int64` `[B]`
* **output:**

  * `processed_signal`: `float32` `[B, 80, T2]` → ví dụ `(1, 80, 151)`
  * `processed_signal_length`: `int64` `[B]`

# Encoder (ConformerEncoder)

* **forward signature:** `(audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None)`
* **input:**

  * `audio_signal`: `float32` `[B, 80, T2]` (tức log-mel chuẩn NeMo)
  * `length`: `int64` `[B]`
* **raw output (như NeMo trả về):**

  * `encoded`: `float32` `[B, 512, T3]` → ví dụ `(1, 512, 19)`  ⟵ **kênh trước, thời gian sau**
  * `encoded_length`: `int64` `[B]`
* **gợi ý chuẩn hoá cho ONNX/runtime:** transpose thành `[B, T3, 512]`.

# Decoder (RNNTDecoder)

* **forward signature:** `(targets, target_length, states=None)`
* **input:**

  * `targets`: `int64` `[B, U]` → ví dụ `(2, 4)`
  * `target_length`: `int64` `[B]` → ví dụ `(2,)`
  * `states=(h0, c0)`: `float32` mỗi cái `[L, B, H]` → với L=1, H=640: `(1, 2, 640)`
* **output (chính xác):** tuple **3 phần tử**

  1. `pred`: `float32` `[B, H, U+1]` → ví dụ `(2, 640, 5)`
     (NeMo RNNT prediction net trả thêm 1 bước so với U)
  2. `target_length_out`: `int64` `[B]`
  3. `states=(h1, c1)`: `float32` mỗi cái `[L, B, H]` → `(1, 2, 640)`
* **gợi ý chuẩn hoá cho ONNX/runtime:** transpose `pred` → `[B, U+1, H]` để khớp `joint`.

# Joint (RNNTJoint)

* **forward signature:** `(encoder_outputs, decoder_outputs, encoder_lengths=None, transcripts=None, transcript_lengths=None, compute_wer=False) -> ...`
* Khi gọi **`forward`** với model này, có cờ `fuse_loss_wer` → sẽ yêu cầu lengths/transcripts để tính loss/wer.
* **Để chỉ lấy logits**, gọi **`model.joint.joint(enc, pred)`** (hàm con pure joint).
* **input (pure joint):**

  * `enc`: `float32` `[B, T, 512]` (đã transpose từ encoder raw)
  * `pred`: `float32` `[B, U, 640]` (đã transpose từ decoder pred)
* **output:**

  * `logits`: `float32` `[B, T, U, vocab]` → ví dụ `(1, 2, 3, 1025)` (vocab = 1025 = 1024 + blank)

---

## Mapping đề xuất khi export ONNX

* **encoder.onnx**

  * inputs: `processed_signal [B,80,T2] (f32)`, `processed_signal_length [B] (i64)`
  * outputs: `encoded [B,T3,512] (f32)` **(đã transpose trong wrapper)**, `encoded_length [B] (i64)`
* **predictor.onnx**

  * inputs:

    * `targets [B,U] (i64)`
    * `target_length [B] (i64)`
    * `states_hc [2,L,B,H] (f32)` với L=1, H=640  (stack của h & c: `[0]=h, [1]=c`)
  * outputs:

    * `pred [B,U+1,640] (f32)` **(đã transpose từ \[B,640,U+1])**
    * `next_states_hc [2,L,B,H] (f32)`
* **joint.onnx**

  * inputs: `enc [B,T,512] (f32)`, `pred [B,U,640] (f32)`
  * outputs: `logits [B,T,U,1025] (f32)`
  * wrapper gọi `self.joint.joint(enc, pred)` (không dùng `forward`).

---

## Ghi chú quan trọng cho runtime/giải mã

* Với greedy step-by-step:

  * dùng `U=1` mỗi bước → lấy `pred[:, 0, :]` (hoặc `[:, 1, :]` nếu muốn bỏ “+1” đầu) để feed vào `joint`.
  * `enc_t`: `[B,1,512]`, `pred_u`: `[B,1,640]` → `logits[:,0,0,:]` → argmax.
* Nếu so khớp đúng “kiểu NeMo”, giữ `U+1`; còn nếu pipeline của bạn đang giả định `U`, có thể cắt `pred = pred[:, 1:, :]`.

Cần mình chỉnh sẵn file exporter theo đúng chuẩn hoá ở trên không? Mình có thể gửi lại một bản `export_rnnt_core_onnx.py` đã patch 3 wrapper (encoder/predictor/joint) theo thông số bạn vừa probe.



=====================================


Đúng rồi: model của bạn **có preprocessor** (AudioToMelSpectrogramPreprocessor) và về lý thuyết **có thể export ONNX**. Nhưng có vài điểm thực tế:

## Ngắn gọn

* **Có thể export toàn bộ “core”** thành ONNX: `encoder.onnx`, `predictor.onnx`, `joint.onnx` (đã OK).
* **Preprocessor (wav→log-mel)**: *có thể* export, nhưng **dễ gãy** tùy phiên bản PyTorch/torchaudio/ONNX vì dính **STFT/complex**. Giải pháp production an toàn nhất là **bỏ export preproc**, tính log-mel ở runtime (torchaudio/librosa) rồi đưa thẳng vào `encoder.onnx` — bạn đã có code này.

## Chi tiết preprocessor

* Preprocessor hiện nhận `input_signal [B,T] (f32)` và trả `processed_signal [B,80,T2] (f32)`.
* Lỗi bạn gặp trước đó là do mình lỡ thêm arg không tồn tại (`use_complex_as_real`). **Bỏ tham số đó** là điều kiện cần, nhưng:

  * Một số bản exporter vẫn *khó chịu* với luồng tính có **complex tensor** (STFT) → export có thể fail tùy môi trường (opset, PyTorch/torchaudio version).
* Nếu bạn **vẫn muốn export preproc**:

  1. Thử lại `torch.onnx.export` **không** dùng `use_complex_as_real`, opset **≥17**, `training=EVAL`, `do_constant_folding=True`.
  2. Nếu vẫn fail, có 2 đường:

     * **Dựng preproc “real-only”**: tự cài STFT bằng conv1d + ma trận DFT thực, power=2, rồi nhân ma trận mel (nn.Linear fixed-weight) → ONNX-friendly.
     * **Trace torchaudio MelSpectrogram** (một số setup export được), nhưng cũng hên xui theo phiên bản.

## Còn “export full end-to-end wav→text” thì sao?

* RNNT decoding có vòng lặp T×U (greedy/beam) → **không đơn giản** để nhốt hết vào **một** file ONNX (cần control-flow). Cách tiêu chuẩn trong sản phẩm:

  * Export **ba khối**: `encoder`, `predictor`, `joint`.
  * Viết **decoding** (greedy/beam) bên ngoài (Python/C++/ONNX Runtime loop).
  * Preproc: tính ngoài ONNX cho **ổn định & dễ kiểm soát**.

## Kết luận thực dụng

* **Có**: preprocessor tồn tại và *có thể* export trong một số môi trường.
* **Khuyến nghị**: giữ kiến trúc “hybrid” đã làm:
  preproc (runtime) → `encoder.onnx` → `predictor.onnx` + `joint.onnx` (loop greedy/beam).
  Đây là cách phổ biến nhất để đem RNNT ra production, tránh phụ thuộc fragile của STFT trong ONNX.

Nếu bạn muốn, mình có thể gửi ngay một biến thể `PreprocWrap` đã chỉnh để thử export lần nữa (không dùng complex arg) + checklist phiên bản (PyTorch/torchaudio/opset) cần để tăng xác suất thành công.



======================================



(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo$ python -m vpb_mod.export.probe_onnx_io /home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy

====================================================================================================
ENCODER: /home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy/onnx/encoder.onnx
====================================================================================================
Inputs:
  [0] name='processed_signal', type=tensor(float), shape=['B', 80, 'T2']
  [1] name='processed_signal_length', type=tensor(int64), shape=['B']
Outputs:
  [0] name='encoded', type=tensor(float), shape=['B', 'T3', 512]
  [1] name='encoded_length', type=tensor(int64), shape=['B']

====================================================================================================
PREDICTOR: /home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy/onnx/predictor.onnx
====================================================================================================
Inputs:
  [0] name='targets', type=tensor(int64), shape=['B', 'U']
  [1] name='states_hc', type=tensor(float), shape=[2, 1, 'B', 640]
Outputs:
  [0] name='pred', type=tensor(float), shape=['B', 'U', 640]
  [1] name='next_states_hc', type=tensor(float), shape=[2, 1, 'B', 640]

====================================================================================================
JOINT: /home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy/onnx/joint.onnx
====================================================================================================
Inputs:
  [0] name='enc', type=tensor(float), shape=['B', 'T', 512]
  [1] name='pred', type=tensor(float), shape=['B', 'U', 640]
Outputs:
  [0] name='logits', type=tensor(float), shape=['B', 'T', 'U', 1025]
(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo$ 
