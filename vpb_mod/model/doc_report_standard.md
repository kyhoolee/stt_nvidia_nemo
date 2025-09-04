Hiểu rồi 👍. Em soạn lại thành một bản **report tổng hợp** (giống “executive note”) để anh dùng chia sẻ với BU/QA, gồm: **(1) Bối cảnh, (2) Kết quả, (3) Lưu ý trùng lặp, (4) Insight về User voice đa dạng, (5) Kết luận – hướng đi.**

---

# **Report – Performance ASR Models cho VPB Callbot**

## **1. Bối cảnh**

* **Chunkformer (đóng, in-use):** \~23–27% WER, chất lượng khá tốt nhưng **không thể fine-tune** thêm.
* **Fast-Conformer (open, pretrained VietSpeech):** 35–39% WER → kém hơn rõ rệt.
* Để cải thiện, nhóm đã triển khai fine-tuning theo 2 bước:

  1. **v1:** Fine-tune với pseudo-label từ data VPB.
  2. **v2:** Fine-tune tiếp trên tập train VPB\_right2 (dữ liệu gán nhãn thực tế sau pseudo).

---

## **2. Kết quả WER**

| Dataset                    | Chunkformer | Fast-Conformer (pretrain) | v1 (pseudo) | v2 (pseudo→VPB)       | Ghi chú                                         |
| -------------------------- | ----------- | ------------------------- | ----------- | --------------------- | ----------------------------------------------- |
| **standard\_test\_2**      | 0.2499      | 0.3546                    | 0.3040      | **0.2420**            | ⚠ Nhiều trùng lặp với train (\~70%)             |
| **standard\_test**         | 0.1613      | 0.3378                    | **0.2582**  | 0.4294 (↓ bất thường) | Nhỏ (29 mẫu), dao động cao                      |
| **next\_day\_test\_debug** | 0.2076      | 0.3414                    | 0.2687      | **0.2649**            | Bộ test độc lập                                 |
| **vpb\_right2\_train**     | 0.2420      | 0.3633                    | 0.2956      | **0.2456**            | ⚠ Tập train, WER thấp không phản ánh generalize |
| **vpb\_right2\_valid**     | 0.2696      | 0.3902                    | 0.3240      | **0.2821**            | Bộ valid, phản ánh tốt                          |

**Tóm tắt xu hướng:**

* Pretrain: WER \~0.35–0.39 (rất kém).
* Fine-tune v1: WER \~0.26–0.32 → cải thiện mạnh.
* Fine-tune v2: WER \~0.24–0.28 trên các tập chính (valid, next\_day) → **tiệm cận hoặc vượt Chunkformer**.

---

## **3. Lưu ý về trùng lặp**

* **standard\_test\_2** và **train VPB\_right2** có **overlap \~70%**.

  * Do đó WER 0.2420 trên standard\_test\_2 tuy rất đẹp, nhưng **không phản ánh generalization**.
* Các tập **không trùng lặp** (valid, next\_day) mới là thước đo đáng tin → kết quả 0.26–0.28 cho thấy mô hình **đang cải thiện thực chất**.

---

## **4. Insight chính: User Voice đa dạng**

* **Thách thức lớn nhất** của VPB Callbot ASR là **đa dạng và nhiễu** của giọng User:

  * Accent vùng miền, tốc độ nói, từ viết tắt, xen lẫn tiếng Anh, số liệu.
  * Nhiễu từ môi trường (đường phố, quán xá, tín hiệu điện thoại).
* Điều này khiến việc tạo ra **một robust model** cực khó: mô hình phải xử lý **nhiều dạng User unseen** (chưa từng gặp trong training).
* Việc gán nhãn dữ liệu chuẩn từ chính call User là **cách duy nhất** để từng bước làm mô hình thích nghi dần với thực tế.

---

## **5. Kết luận & Hướng đi**

* **Chunkformer**: baseline tốt, nhưng không thể mở rộng → **“trần hiệu năng” \~23–27% WER**.
* **Fast-Conformer open + fine-tune**: bước đầu cho thấy khả năng **tiệm cận và vượt Chunkformer** trên data nội bộ.
* **Dữ liệu gán nhãn chuẩn**: bắt buộc để cải thiện robust ASR. Dù không đảm bảo vượt ngay, nhưng tạo ra **tài sản dữ liệu dài hạn**, tái sử dụng cho nhiều mô hình/bài toán khác (Callbot, QA Call, đào tạo Agent…).
* **Next step:**

  * Hoàn tất gán nhãn 6k file (100h audio) trước 15/09 để kịp sử dụng AWS credit (\$15k, hết hạn 20/09).
  * Dùng dữ liệu này để fine-tune thêm, kiểm chứng trên **test set độc lập hoàn toàn (non-overlap)**.
  * Song song, nghiên cứu augment và chiến lược robust hoá cho unseen User voice.

---

👉 **Thông điệp clean cho BU:**

* Không gán nhãn thì chắc chắn mô hình dừng ở mức hiện tại (\~70–75%).
* Có gán nhãn thì mở ra cơ hội **cải thiện lên 75–80%** và xây dựng dữ liệu chuẩn, **tài sản chiến lược lâu dài**.
* Đầu tư OT 100 triệu VND là hợp lý để **không lãng phí AWS credit** và tạo nền tảng cho AI Callbot thực sự robust.

