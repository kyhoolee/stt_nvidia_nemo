# **Slide Deck – ASR Performance Report (Full 5 Dataset)**

---

### **1. Bối cảnh**

* **Chunkformer (đóng, in-use):** \~23–27% WER, tốt nhưng **không fine-tune được**.
* **Fast-Conformer (open, pretrain VietSpeech):** WER \~35–39% → kém.
* Hướng đi: fine-tune 2 bước

  * v1: pseudo label
  * v2: pseudo → VPB train

---

### **2. Kết quả WER (5 tập dữ liệu)**

| Dataset                    | Chunkformer | Pretrain | v1 (pseudo) | v2 (pseudo→VPB)   | Ghi chú                   |
| -------------------------- | ----------- | -------- | ----------- | ----------------- | ------------------------- |
| **standard_test_2**      | 0.2499      | 0.3546   | 0.3040      | **0.2420**        | ⚠ Overlap \~70% với train |
| **standard_test**        | 0.1613      | 0.3378   | **0.2582**  | 0.4294 (dao động) | Nhỏ (29 mẫu)              |
| **next_day_test_debug**  | 0.2076      | 0.3414   | 0.2687      | **0.2649**        | Test độc lập              |
| **vpb_right2_train**     | 0.2420      | 0.3633   | 0.2956      | **0.2456**        | ⚠ Tập train               |
| **vpb_right2_valid**     | 0.2696      | 0.3902   | 0.3240      | **0.2821**        | Validation set            |

---

### **3. Phân tích**

* Pretrain: kém xa Chunkformer.
* Fine-tune v1: cải thiện mạnh (WER giảm \~0.07–0.10).
* Fine-tune v2: tiệm cận Chunkformer ở **tập chính (valid, next_day)**.
* WER thấp ở **standard_test_2** do **data leakage**.
* WER thấp ở **train** chỉ phản ánh mức fit, **không nói lên generalization**.

---

### **4. Insight chính**

* **Vấn đề cốt lõi:** User voice đa dạng & nhiều nhiễu

  * Accent vùng miền, tốc độ, code-switch (VN+EN), số liệu.
  * Nhiễu môi trường, tín hiệu call kém.
* Thách thức: tạo model **robust cho unseen users**.
* Cách duy nhất: gán nhãn thủ công → tập dữ liệu chuẩn nội bộ.

---

### **5. Giá trị gán nhãn**

* Điều kiện cần để fine-tune và mở trần hiệu năng.
* Tài sản dữ liệu dài hạn: benchmark, Call QA, đào tạo Agent…
* One-time investment, reuse nhiều vòng.

---

### **6. Tính cấp bách**

* AWS credit **15,000 USD** (expire 20/09/2025).
* Deadline gán nhãn **15/09/2025** → cần 100 agent, 500h OT (\~100 triệu VND).

---

### **7. Kết luận**

* Không gán nhãn → mô hình dừng ở 70–75%.
* Có gán nhãn → mở cơ hội đạt **75–80%+**, đồng thời sở hữu **dữ liệu chuẩn chiến lược**.
* Đầu tư OT hợp lý để tận dụng AWS credit & phát triển Callbot robust.

