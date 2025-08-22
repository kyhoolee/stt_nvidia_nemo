## Exploration cost 
- Chi phí 1h là 30$
- Tổng tiền là 10k$ -> có tổng 300h để training 
- Với dữ liệu 2.5mil sample / ~ 2500h -> train 1 epoch mất 1h30p 
- Training với 2500h data chỉ có khoảng 200 epochs 
- Model có thể hội tụ sau khoảng 30 epoch ? 

## Technique issue 
1. Vocab -> tối cường issue để handle việc out-of-scope dataset 
-> xuất hiện nhiều từ mới -> cách phát âm mới lạ (giọng địa phương) -> buộc phải nắm cách xử lý 
- How to incremental extend vocab
- Or prepare a large-size of vocab instead only training data vocab 

2. Noise handle -> tối cường issue của normal-user qua telephony
-> nhiều nền có thể từ lớn đến khá lớn -> có thể có bộ phân loại là ko nghe rõ ko ? -> thay vì STT 1 cách blindly 

3. Transfer-learning -> tối cường technique -> nhưng ở câp độ cao hơn 
-> Liệu có thể combine nhiều training weight lại 1 cách hợp lý khi phải training nhiều loại dataset ko ?

4. Grid-search -> tối cường issue của chất lượng vs chi phí 
-> Liệu có cách tìm ra cấu hình model hợp lý với chi phí rẻ nhất có thể ko 

5. RL-agent for using resource optimally -> tối cường issue của model exploration 