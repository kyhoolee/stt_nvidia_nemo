# 00 — INDEX: kế hoạch đào sâu kiến trúc ASR

Kế hoạch chia kiến thức ASR thành các file con, mỗi file đào sâu một thành phần của pipeline theo một template thống nhất.
Đây là tài liệu kế hoạch; nội dung chi tiết nằm ở các file con `0N_*.md` sẽ viết sau khi duyệt.

---

## 1. Template chuẩn cho mỗi file con

Mỗi file con trình bày một thành phần theo đúng khung sau:

- **Vai trò** — thành phần này làm gì trong pipeline, đặt ở đâu.
- **Input** — kiểu dữ liệu, shape, dtype, ý nghĩa từng chiều.
- **Output** — kiểu dữ liệu, shape, dtype, ý nghĩa.
- **Bộ xử lý ở giữa** — các phép biến đổi, công thức, tham số chính.
- **Flow** — luồng dữ liệu đi qua thành phần (kèm sơ đồ Mermaid).
- **Độ phức tạp** — chi phí tính toán/bộ nhớ theo độ dài chuỗi, batch, kích thước model.
- **Cách đánh giá chất lượng** — đo thành phần này tốt/xấu bằng chỉ số gì.
- **Neo mã nguồn** — file và class thật trong NeMo / `vpb_mod`.
- **Glossary** đầu file và **Tự kiểm nhanh** cuối file (theo chuẩn doc chung).

---

## 2. Danh sách file con

Pipeline ASR đi từ âm thanh thô tới văn bản. Mỗi mắt xích là một file con.

Ba kiểu giải mã CTC / RNNT / AED tách thành ba file riêng; mỗi file tự chứa cấu trúc + hàm mất mát + cách giải mã của kiểu đó.

| Thứ tự | File | Thành phần | Input → Output | Cần chạy code |
| --- | --- | --- | --- | --- |
| 1 | `01_pipeline_overview.md` | Toàn cảnh pipeline | Audio → Text | Không |
| 2 | `02_tokenizer.md` | Tokenizer SentencePiece BPE | Text ↔ token id | Tùy chọn |
| 3 | `03_audio_to_mel.md` | Tiền xử lý audio → log-mel | Waveform → log-mel [B,80,T] | Tùy chọn |
| 4 | `04_specaugment.md` | Tăng cường dữ liệu (chỉ khi train) | log-mel → log-mel có mask | Không |
| 5 | `05_encoder_conformer.md` | **Encoder Conformer / Fast-Conformer (trung tâm)** | log-mel → biểu diễn ẩn [B,T,512] | **Có** |
| 6 | `06_decode_ctc.md` | Giải mã CTC (cấu trúc + loss + decode) | encoder out → token | Tùy chọn |
| 7 | `07_decode_rnnt.md` | **Giải mã RNNT + Joint (model VPB dùng)** | encoder out → token | Tùy chọn |
| 8 | `08_decode_aed.md` | Giải mã AED encoder–decoder | encoder out → token | Tùy chọn |
| 9 | `09_evaluation_wer.md` | Đánh giá chất lượng (WER) | pred + ref → WER | Không |

- **Ghi chú** — các chiến lược giải mã chung (greedy, beam, streaming cache-aware) trình bày trong `07_decode_rnnt.md` (kiểu VPB dùng) và tham chiếu từ hai file CTC/AED để tránh lặp.

---

## 3. Cấu trúc riêng cho file trung tâm `05_encoder_conformer.md`

File này đáp ứng yêu cầu bóc tách kiến trúc chi tiết. Bố cục dự kiến:

- **Danh sách layer đầy đủ** — liệt kê từng layer của encoder theo thứ tự, từ subsampling tới layer cuối.
- **Phân biệt Conformer và Fast-Conformer** — khác nhau ở mức subsampling (4× so với 8×) và hệ quả về độ dài chuỗi, chi phí.
- **Khối được lặp lại** — chỉ rõ đoạn nào lặp (ở model VPB là 17 lần `ConformerLayer`).
- **Cấu trúc phần tử lặp** — bóc `ConformerLayer` thành bốn module con theo cấu trúc macaron:
  - FeedForward thứ nhất (hệ số 1/2).
  - Self-attention có mã hóa vị trí tương đối (RelPositionMultiHeadAttention).
  - Convolution module (pointwise → depthwise causal → BatchNorm → Swish → pointwise).
  - FeedForward thứ hai (hệ số 1/2) và LayerNorm cuối.
- **Ý nghĩa từng module** — vì sao cần kết hợp convolution (bắt quan hệ cục bộ) với self-attention (bắt quan hệ toàn cục).
- **Số tham số** — số tham số từng phần và toàn encoder (lấy bằng cách chạy code, xem Mục 4).

---

## 4. Cách lấy số liệu kiến trúc chính xác

Yêu cầu in ra danh sách layer thật. Có hai cách, đề xuất làm theo thứ tự.

- **Cách 1 — đọc nguồn và bản dump sẵn có (rẻ, không cài đặt)**:
  - Bản in cấu trúc model thật đã có sẵn trong repo: `vpb_mod/export/issue.md` và `vpb_mod/export_direct/asr_model.md` (in từ `print(asr_model)` trên model VPB đã train).
  - Đọc mã nguồn `nemo/collections/asr/modules/conformer_encoder.py` để giải thích từng module.
  - Đủ để liệt kê layer và giải thích cấu trúc; thiếu số tham số chi tiết từng module.
- **Cách 2 — cài lib và chạy để lấy số tham số (đắt hơn, cần môi trường)**:
  - Tài liệu VPB tham chiếu môi trường conda tên `nemo`; cần xác nhận môi trường này còn dùng được.
  - Dựng model từ config `tutorials/asr/configs/fast-conformer_transducer_bpe.yaml` rồi gọi `model.summarize()` để in số tham số từng module.
  - Cho số liệu đầy đủ nhất nhưng phụ thuộc cài đặt nặng (PyTorch, NeMo).

Đề xuất: làm Cách 1 trước (đã có bản dump thật), bổ sung Cách 2 khi viết `05_encoder_conformer.md` nếu môi trường sẵn sàng.

---

## 5. Thứ tự thực hiện (đã chốt: tuần tự từ file 1)

- **Giai đoạn A — khung pipeline**: file 1 (overview) để có bản đồ trước.
- **Giai đoạn B — đầu vào**: file 2 (tokenizer), file 3 (audio → mel), file 4 (specaugment).
- **Giai đoạn C — lõi model**: file 5 (Conformer, trung tâm).
- **Giai đoạn D — giải mã**: file 6 (CTC), file 7 (RNNT, trọng tâm), file 8 (AED).
- **Giai đoạn E — đánh giá**: file 9 (WER).

Mỗi file viết xong sẽ báo cáo và xin phép trước khi sang file kế tiếp (theo quy trình scope tuần tự).

---

## 6. Quyết định đã chốt

- **Điểm bắt đầu** — viết tuần tự từ file 1.
- **Tách giải mã** — CTC / RNNT / AED thành ba file riêng (file 6–8).
- **Cách lấy số liệu kiến trúc** — dùng Cách 1 (bản dump thật sẵn có trong repo + đọc source); chỉ chạy `summarize()` (Cách 2) ở cuối nếu cần số tham số từng module.
  - Cơ sở: bản dump `vpb_mod/export/issue.md` đã chứa nguyên cây layer của model thật; mã nguồn Conformer gom gọn trong `conformer_encoder.py` và `conformer_modules.py`, không cần trace import nhiều tầng.
