🚀 Bước 1: Khởi tạo - Tải model và các thành phần phụ trợ...
[NeMo I 2025-09-23 08:55:03 mixins:181] Tokenizer SentencePieceTokenizer initialized with 1024 tokens
[NeMo I 2025-09-23 08:55:03 features:305] PADDING: 0
[NeMo I 2025-09-23 08:55:04 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0}
[NeMo I 2025-09-23 08:55:04 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0}
[NeMo I 2025-09-23 08:55:04 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0}
[NeMo I 2025-09-23 08:55:05 save_restore_connector:275] Model EncDecRNNTBPEModel was successfully restored from /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo.
   - Tokenizer: Vocab size = 1025, Blank ID = 1024
   - Predictor LSTM: 1 layer(s), Hidden size = 640
   - Chế độ thực thi: CPU (CPUExecutionProvider)
⚡ Tải 3 model ONNX vào Inference Session...
✅ Tải xong model và các thành phần.

🚀 Bước 2: Bắt đầu đánh giá trên file manifest: /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl
🔍 Running ONNX transcription and WER calculation...

--- Processed 1/29 samples. ---
Sample:
  REFERENCE: alo ạ anh ro him đang nghe máy hả anh em chào anh em là đức bên ngân hàng vp banh ạ bên ngân hàng gọi thông báo sớm khoản vay ô tô với khoản vay tín chấp ấy sắp đến hạn thanh toán là ngày mười lăm tháng hai anh nhận được tin nhắn chưa ạ ờ mười bảy triệu sáu trăm năm mươi nghìn ý thì anh để ý thanh toán đúng hạn giúp em anh nhá để tránh phát sinh kỳ lãi phạt với ảnh hưởng đến lịch sử tín dụng ấy ạ dạ vâng ạ dạ rồi em cảm ơn anh nhá vâng em chào anh ạ
  PREDICTED: anh anh à em anh anh đức v bên gọi gọi thông thông cái khoản ô ô khoản chấp chấp sao cho hạn thanh là ngày mười tháng tháng anh anh nhận được chưa chưa à mười bảy sáu năm nghìn thì anh ý thanh đúng giúp anh anh để để để để phát phát là ph ảnh lịch lịch t dụng v cảm anh chào chào

--- Processed 2/29 samples. ---
Sample:
  REFERENCE: alo đúng rồi vậy anh em dạ rồi ạ ok ạ rồi dạ rồi
  PREDICTED: đúng

--- Processed 3/29 samples. ---
Sample:
  REFERENCE: tổng đài viên ngân hành vi bi banh xin nghe anh chị kết nối đến chuyên viên hỗ trợ hoa có thể hỗ trợ gì cho anh chị không ạ em hỏi đi anh gì ơi nói chuyện cho lịch sự vào sử dụng những cái từ ngữ cho lịch sự văn minh văn hóa lên
  PREDICTED: kết kết hồ h có h h trợ gì chị anh anh ơi ơi nói nói lịch như lịch lên

--- Processed 4/29 samples. ---
Sample:
  REFERENCE: bác tổng đài mày bán lồn à hay bán vòng hoa hả đây là nhà tao đang bán đại lý vòng hoa này địt mẹ mày bố mẹ mày vừa chết đấy nhanh lên bố mẹ mày vừa bị ô tô đâm chết đấy cái địt mẹ mày cái con chó ngân hàng địt mẹ mày toang lồn làm gì nhà chúng mày mà chúng mày gọi vào máy tao địt mẹ mày địt cả họ nhà mày chúng mày đến hẳn nhà tao đây này nhá nhà tao chưa thay đổi địa chỉ địt con mẹ cả họ ngân hàng nhà mày địt con mẹ cái bọn địt mẹ cái ngân hàng chó bố mày còn tồn tại địt mẹ là cái lũ ngân hàng nhà mày bố mày nợ đến hẳn nhà bố mẹ mày mấy con chó thu phí hết chưa địt cả họ nhà chúng mày
  PREDICTED: bán bánn à bán đây đang bán bán đại này chết chết chết ô đ mẹ con con con vàot nhà nhà chúng nhà nhà tao tao này nhà nhà con con con hàng nhà nhà con con cái cái mày cái chó chó chó chó bố con cái cái nhà nhà nhà bố bố nhà bố bố con con nhà nhà

--- Processed 5/29 samples. ---
Sample:
  REFERENCE: số điện thoại của anh bùi trọng hưng đúng không ạ chào anh em là ngân hàng vi bi banh ý trao đổi với anh khoản vay thế chấp và tín chấp khi này đến hạn thanh toán rơi vào mười ngày nữa mười chín anh nhận thông báo từ phía ngân hàng chưa anh chín triệu ba ạ dạ vâng thế có gì anh cứ để ý thanh toán đúng hạn giúp em rơi vào ngày ờ mười chín giúp em nhá do khoản này của anh trước đấy nó đã có cái phần cơ cấu nợ ở hồi cô vít rồi ý ạ nên là ngân hàng có yêu cầu thanh toán đúng hạn trong trường hợp phát sinh quá hạn thì khoản vay sẽ bị chuyển lên nhóm nợ bốn từ ngày đầu tiên quá hạn đồng thời rằng sẽ bị chấm dứt hợp đồng vay trước hạn và có thể khởi kiện ý không ạ ý là em vầng thế có gì anh cứ để ý lịch đúng hạn giúp em là được anh nhá tránh phát sinh phí phạt của anh là chín triệu ba trăm nghìn em cảm ơn chào anh
  PREDICTED: trọng trọng đúng em ng tra với khoản khoản thế thế cái t t thanh mười nữa nữa mười mười anh anh anh thông từ phía anh triệu triệu ạ dạ có gì để để thanh thanh đúng giúp ngày ngày ngày mười mười giúp nh khoản khoản của của anh trước nó có có phần cú một nên yêu thanh thanh đúng đúng trường phát phát sinh v sẽ chuyển chuyển bốn từ ngày hạn đồng đồng là là bị bị các hợp hợp có có kh kh kiện kiện ý ý có cứ giúp là được phát phát ph ph chín chín ba nghìn em cảm chào

--- Processed 6/29 samples. ---
Sample:
  REFERENCE: đúng rồi ờ em ơi ờ cái nào anh cũng thanh toán ờ bình thường đầy đủ mà em anh có chịu đâu ừ em ơi em em mới tám năm nay anh chưa bao giờ sai em ơi em đang nói như thế là như nào bảy năm nay sáu bảy năm nay rồi anh đã sai đâu ok em
  PREDICTED: anh bình đầy đầy mà chịu đâu giờ em em em lại bảy anh không đâu

--- Processed 7/29 samples. ---
Sample:
  REFERENCE: alo em chào chị ạ cho em hỏi đây có phải số điện thoại của chị phạm thị thu thủy không ạ em là linh em gọi chị từ ngân hàng vi bi banh em gọi điện để thông báo là chị đang có một khoản vay tín tiếp trễ hạn một ngày số tiền cần thanh toán là hai triệu đồng ạ lý do vì sao chưa thanh toán cho ngân hàng vậy chị thế không vay ạ khoản vay này đang đứng tên của chị mà thế có phải là chị phạm thị thu thủy ờ sinh năm một nghìn chín trăm sáu tám không ạ thế lại không vay hả chị khi thế có khoản vay của ngân hàng thì chị phải biết chứ thế sao bây giờ lại bảo không vay ạ sử dụng tiền của ngân hàng sao lại bảo không vay ạ chị đang có khoản vay tín chấp của ngân hàng đây ạ em gọi điện để nhắc chị thanh toán cái khoản vay này thế bao giờ có kế hoạch thanh toán cho ngân hàng đây ạ nếu mà chị không thanh toán thì để càng lâu phát sinh lãi và phí phạt càng nhiều đấy ngân hàng không hỗ trợ được đâu ạ bây giờ vay lại bảo không vay là sao ạ không vay thì ra chi nhánh ngân hàng để làm việc đi ạ hay lừa đảo thì báo công an đi chị lừa đảo thì báo công an đi còn nếu mà chị cứ để tình trạng như thế này chị ơi lừa đảo hay không ý bản thân chị bản thân chị là người rõ nhất thì ạ còn nếu mà chị bảo là lừa đảo thì chị báo công an ạ chị nhá còn nếu mà không thanh toán cho ngân hàng không có kế hoạch cụ thể em ghi nhận thông tin là từ chối thanh toán mọi phát sinh lãi phí phạt rủi ro chị tự chịu trách nhiệm ngân hàng không hỗ trợ được đâu ạ cuộc gọi vẫn tiếp tục đổ lại cho chị cuộc gọi sẽ tiếp tục đổ lại cho chị và người thân người tham thiếu bạn bè đồng nghiệp của chị lúc đấy ảnh hưởng tới cả uy tín danh dự của chị ngân hàng không chịu trách nhiệm chị nhá em ghi nhận thông tin là từ chối thanh toán ạ em xin lừa đảo hay không thì chị báo công an sẽ rõ nhá em ghi nhận thông
  PREDICTED: cho đây số thoại của phạm thu thủ không ⁇  em cho chị vi em em điện điện thông là chị chị có một t t tr một một số cần thanh hai đồng đồng ạ ạ vì sao chưa cho cho vậy vậy không không khoản khoản này này đang đứng của mà mà có là thị phạm phạm thu thủ  sinh sinh chín chín tám tám không không không không hả hả chị chị khoản của của ng chị phải phải chứ bây bây lại bảo bảo không v ạ sử tiền tiền ng hàng sao bảo v v ạ đang có t t của ng em để để nhắc nhắc thanh thanh cái khoản này này bao bao ng thế thế thế chị không toán để để để lâu lâu sinh l l và ph càng càng ng ng không được đâu bây v lại bảo bảo v v là là ạ không ra ra chi ng ng hàng hàng làm đi ạ l l đảo chị công công đi đi l l đảo báo công đi đi đi nếu chị chị cứ tình tình như này chị ơi ơi l l đảo đảo không bản bản chị chị bản bản bản thân là chị còn còn bảo bảo l l đảo đảo công ạ nh còn còn cho ng không kế cụ cụ em nhận thông là từ từ từ thanh ạ phát phát phát l ph ph giao tự tự trách trách ng ng h ạ cuộc vẫn vẫn tiếp tiếp đ đ lại lại chị cuộc cuộc sẽ sẽ tiếp tiếp đ đ lại cho người người người tham tham bạn bạn đồng đồng của của lúc lúc ảnh hưởng cả   danh danh của của ng không không chịu trách chị nh em em thông là từ thanh thanh ạ l l đảo không không chị công sẽ rõ rõ nh em thông thông

--- Processed 8/29 samples. ---
Sample:
  REFERENCE: đúng em ủa chị có vay bên ngân hàng em đâu mà đóng đúng rồi nhưng mà chị không có vay ngân hàng này ủa chị vay chị tự biết chứ sao lại không biết được hả chị vay chị phải tự biết chứ sao không biết không cái đó chị không không chị không có vay chị không có trả ờ thì bây giờ lừa đảo thiếu cha gì chị có vay đâu mà bảo chị vay ha thôi nha ừ chứ còn tao không có vay nha tụi mày đừng có ăn đi lừa đảo nha đi lừa đảo mấy người dân đồ tội nghiệp ra nha đúng rồi đúng rồi rồi đúng rồi tao không có nợ ngân hàng gì hết trơn á bọn mày lúc đó bảo bọn mày là bọn lừa đảo ai tin bọn mày hả thôi nha
  PREDICTED: đúng bên đúng đúng nhưng không v ng ng hàng chị chị thì biết sao không không được được chị chị không không không không không không chị chị trả bây l l đảo chị v chị chị chứ chứ tao không nha nha nha đừng đi l l nha nha đi đi l mấy mấy đúng đúng đúng đúng đúng đúng tao ng ng hết rồi rồi lúc lúc bọn bọn là bọn bọn ai bọn bọn hả thôi thôi

--- Processed 9/29 samples. ---
Sample:
  REFERENCE: alo anh thanh nghe máy ạ à em chào anh ạ em là anh thanh em máy đúng không ạ vâng em chào anh ạ em là mi liên hệ với anh từ ngân hàng vi pi banh anh có cái khoản vay thế chấp bên em đến ngày mười lăm đến hạn thanh toán em làm tròn là hai mươi triệu chín trăm đấy có gì anh để ý thanh toán đúng hạn giúp em anh nhá vâng à thế ạ vâng bởi vì là cái khoản này của anh nên thấy đã được cơ cấu nợ rồi thế nên là cần phải thanh toán đúng hạn có gì anh báo với bên chủ đầu tư thanh toán đúng hạn giúp em bởi vì trong trường hợp phát sinh quá hạn một ngày thôi thì khoản vay cũng sẽ bị chuyển tối thiểu lên nợ nhóm bốn và vi bi banh sẽ yêu cầu tất toán trước hạn cũng có thể cũng còn cũng như là khởi kiện ra tòa án để xử lý đấy ạ thế anh báo với bên chủ đầu tư để ý đến ngày thanh toán đúng hạn giúp em nhá cái này là quy định của bên ngân hàng thôi ạ a lô
  PREDICTED: anh nghe ạ anh anh anh nghe đúng em em em em liên với với anh hàng anh anh cái khoản thế bên em đến ngày ngày hạn hạn thanh em hai hai chín chín chín ý ý để ý toán đúng giúp giúp anh à à à à ạ ạ bởi là khoản khoản của em đã được cơ rồi rồi nên nên cần cần thanh thanh đúng đúng có với bên tư thanh đúng giúp bởi bởi trong trong phát phát quá quá hạn ngày thì khoản khoản cũng sẽ chuyển chuyển lên nhóm đó đó bốn bốn yêu yêu tất hạn cũng cũng không không cũng cũng như kh kh để lý lý lý đấy đấy ạ anh với với đầu tư để ngày ngày thanh đúng giúp nh cái cái là quy quy của bên thôi

--- Processed 10/29 samples. ---
Sample:
  REFERENCE: alo alo đúng em ơi rồi anh biết rồi cái này bên nô va nó trả tháng nào nó cũng trả
  PREDICTED: tháng tháng nào cũng trả

--- Processed 11/29 samples. ---
Sample:
  REFERENCE: alo anh cường đang nghe máy phải không ạ em chào anh em là thảo gọi từ ngân hàng vi bi banh ạ mình đang có ở ngân hàng em khoản vay ô tô này với hai khoản tín chấp đến hạn thanh toán trong bốn ngày nữa ạ thẻ tín dụng thì ngày mai tới hạn dạ tổng là mười lăm triệu sáu trăm nghìn anh nhá mấy khoản này mình thanh toán đúng hạn được không ạ em cảm ơn vâng vâng ạ vâng ạ dạ vâng anh thường chuyển khoản hay nộp tiền mặt vậy ạ dạ vâng bởi vì khoản vay của anh đã được cơ cấu nợ nên là mình lưu ý thanh toán đúng hạn giúp em nhá trường hợp vị phát sinh quá hạn thì khoản vay sẽ chuyển tối thiểu lên nhóm nợ bốn kể từ ngày đầu tiên quá hạn đấy anh ạ đồng thời ngân hàng có quyền chấm dứt cho vay trước hạn và có thể khởi kiện ra tòa nhân dân có thẩm quyền để xử lý nợ nên mình thu xếp thanh toán đúng hạn nhé em cảm ơn em chào anh ạ vâng dạ chào anh
  PREDICTED: cư ạ ạ em gọi gọi ng thế thế có em em khoản khoản ô ô này hai hai thanh thanh trong ngày ạ th t động ngày ngày ma tới dạ tổng tổng lăm sáu nghìn khoản này mình đấu đúng được được không em cảm v ạ v ạ chuyển hết hết mặt rồi ạ dạ bởi bởi khoản của của anh đã được cơ n nên mình mình thanh thanh đúng giúp giúp nh nh trường hợp phát quá quá thì thì khoản sẽ sẽ chuyển lên lên nhóm nhóm bốn kể từ từ đầu đấy anh anh đồng đồng ng ng có thống thống ch hạn và kh kh để để xử xử n mình thu thanh thanh đúng em em em anh v v dạ chào chào

--- Processed 12/29 samples. ---
Sample:
  REFERENCE: a lô ừ đúng rồi ừ ok ok ok ok ok ok mọi người ừ anh cảm ơn ok anh chuyển khoản ừ ừ ừm ừ ừ ok anh cảm ơn ok
  PREDICTED:     cảm

--- Processed 13/29 samples. ---
Sample:
  REFERENCE: a alo số điện thoại của chị hằng ạ em chào chị em là trang bên ngân hàng vi bi banh ờ chị hằng ơi em gọi điện báo trước hạn cho chị hằng có khoản vay sắp đến hạn tất toán ấy tức là kế hoạch của mình vẫn là vay lại đúng không chị rơi vào ngày mùng ba ngày ra tết ấy chị ạ ba trăm bảy mươi hai triệu tám trăm có gì chị liên hệ cán bộ chi nhánh vay lại thì làm hồ sơ thì giúp em tất toán đúng hạn tránh phát sinh lãi phạt cao chị nhá một phẩy năm mà lãi trong hạn tính trên tổng toàn bộ số tiền chị tất toán ý à tức là làm hồ sơ rồi kí rồi hôm đấy đẩy nguồn tiền vào thôi đúng không chị nguồn tiền vẫn là kinh doanh rồi đúng không chị vâng có gì chị thu xếp đủ nguồn tiền tất toán đúng hạn giúp em vào hôm đấy nhá em cảm ơn chị ạ em chào chị ạ
  PREDICTED: h em em em chị trang hàng gọi gọi hạn trường trường đang khoản khoản khoản của của tất tất tất kế kế của vẫn v v lại lại vào vào ngày ngày m ba ra ra tế tế chị ạ trăm hai hai tám có có chị liên liên cá cá chi chi v lại tất đúng đúng ph cao cao ph nam l trên toàn số số chị ý à à là là ký rồi hôm đẩy đẩy nguồn nguồn vào đúng nguồn nguồn vẫn vẫn vẫn kinh kinh rồi đúng thu thu nguồn tất tất đúng đúng vào hôm nh nh em cảm cảm chị chị em ạ

--- Processed 14/29 samples. ---
Sample:
  REFERENCE: alo vâng vâng ạ vâng ạ em em em làm hồ sơ rồi đang làm hồ sơ rồi vâng vâng em đang làm hồ sơ rồi ạ vâng vâng vâng vâng vâng vâng vâng
  PREDICTED: em em emng em em

--- Processed 15/29 samples. ---
Sample:
  REFERENCE: a alo ạ cho em hỏi có số điện thoại của chị huệ không ạ à chị huệ em là mai anh ở bên vi bi banh ý chị thì em đang thấy là chị đang có cái khoản nợ xấu bên tổ chức tín dụng khác thì bên vi vi banh bọn em và đang yêu cầu chị đã tất tán toàn bộ các khoản bên vi bi banh tổng dư nợ tám tính hiện tại là gần một trăm bốn mươi mốt triệu và cái khoản này thì chị cũng đang chậm trả hai tháng rồi thì không biết là có lý do gì mà chị lại để chậm trả thế hả chị chị gặp khó khăn gì về tài chính à làm sao hả chị thế là chị tất toán chưa hả chị thế chị đã ra chi nhánh chưa chị ra chi nhánh chưa chi nhánh xa thế thì chị có thanh toán qua thẻ không thế trước đây chị thanh toán qua cái gì đấy thì chị có chuyện qua cái thẻ ấy thôi chị ơi em đang trao đổi với chị sao lại mày tao em đang trao đổi với chị tìm phương án mà sao lại mày tao chị ơi em đang trao đổi lịch sử với chị đừng có bày tao
  PREDICTED: cho có của h không em anh ở bên ba chị chị em thấy chị có n xấu bên tổ t t bên bênb bọn bọn yêu yêu chị tất toàn toàn khoản bên tổng tổng d hiện là gần gần mươi mà cái này này chị cũng đang hai rồi chị chị không là có lý gì chị để để ch trả hả chị chị chị chị chị toán thế thế đã đã ra chi chi chưa chưa chị chi chưa chưa thế thế thế chị thanh thanh qua th th không trước chị thanh thanh còn thì thì cứ chuyển cái cái thôi thôi ơi ơi đang với sao lại em tra tra với chị phương phương này chị em em tra lịch với chị này

--- Processed 16/29 samples. ---
Sample:
  REFERENCE: ờ ờ ờ ờ ờ bên em làm ăn buồn cười lắm chị bảo chị tất toán từ hôm nọ giờ nhưng chả ai hỗ trợ gì cả xong bây giờ lên tới tận ngần ý đấy chưa chả có ai hỗ trợ cả chưa chi nhánh xa lắm có thẻ thẻ bây giờ không nhớ thẻ à không có giấy tờ thì cứ thế chuyển mày bị làm sao phải không không phải dài cái mồm ra thế đâu tí nữa kết bạn gia lô rồi có gì nhắn tin sau nhá
  PREDICTED: bên bên ăn buồn buồn lắm lắm từ hôm hôm nhưng ch ai h xong lên lên tận đấy đấy chưa chưa ai h cả chưa xa xa lắm không không không không chuyển không khôngii thế đâu đâu đâu có nh nh

--- Processed 17/29 samples. ---
Sample:
  REFERENCE: a alo ạ cho em hỏi có phải số điện thoại của anh thùy không ạ dạ vâng em chào anh em là mai anh ở bên vi bi banh anh ạ anh ơi em đang thấy hồ sơ của anh lần này hiện tại thì đang nợ xấu bên tổ chức tín dụng khác đấy anh thì bên em mới đang yêu cầu anh và tất toán toàn bộ khoản bên em tổng dư nợ hiện tại là gần tám mươi lăm triệu anh ạ thì chưa bao gồm lãi phí phạt phát sinh cũng như là dư nợ trả góp đấy thì không biết là có lý do gì ạ để nợ xấu thế hả anh tại vì cái khoản thẻ khoản vay bên em ấy cũng đã thấy chậm trả một tháng rồi đây này hả anh hôm qua anh nói là bao nhiêu tức là anh mới đóng cho bên em khoản là khoản quá hạn thôi đúng không thế còn bên tổ chức anh ơi anh ơi trao đổi lịch sự trao đổi lịch sự mình em đang hỏi anh cơ mà em đã nói gì đâu thế bên tổ chức anh ơi thế bên tổ chức tín dụng khác thì anh đã có kế hoạch thanh toán thế nào chưa đang nợ xấu thì thế nào rồi ạ này anh ơi đừng mày tao với em em đang trao đổi với anh đừng mày tao với em anh này anh đọc này anh đọc lại cái có nghe em nói không có nghe em nói không đang bảo lãnh đang nợ xấu bên kia kìa vi ai bi đồng bộ thì bên em yêu cầu anh phải thất toán toàn bộ anh có đọc được hợp đồng vay không thế anh có đọc hợp đồng vay không này hợp đồng vay là anh đang nợ xấu một bên khác thì bên em yêu cầu anh phải tất toán toàn bộ nhá này anh ơi trao đổi rất là mất lịch sự này trao đổi rất là mất lịch sự thôi nhá không trao đổi lễ phép nên bên em ngắt máy trước liên hệ lại anh sau chào anh
  PREDICTED: cho em có thoại của thù thù không không em anh em anh ở bên vi anh anh anh anh ơi em em hồ hồ của anh này hiện hiện đang đang kh tổ t t dụng đấy anh bên đang yêu yêu và và tất toàn toàn khoản bên tổng tổng hiện là là gần anh anh ạ chưa chưa l phát phát có như là d trả trả đấy đấy mình không không có lý lý gì để để n anh anh cái th th khoản khoản bên cũng cũng ch cả một rồi rồi này hả hả anh hôm hôm anh là bao tức tức anh mới cho bên khoảng quá quá thôi thôi đúng đúng bên bên anh anh anh anh tra tra lịch lịch tra tra lịch em hỏi hỏi anh em em gì gì hết bên anh anh hết bên tổ t t dụng thì anh kế kế thanh thanh chưa n n thế rồi ạ mày tao em em tra với đừng đừng tao tao anh anh anh anh này anh lại có có em không không em không đang bảo anh n k ạ thi bộ bộ thì thì bên yêu phải tất toàn toàn anh có độc có hợp không không anh anh hợp hợp không không hợp hợp là anh n một khác thì em em yêu anh tất toàn toàn anh tra rất rất bất này tra rất rất bất không không để lênt máy liên liên với anh chào chào chào

--- Processed 18/29 samples. ---
Sample:
  REFERENCE: đúng rồi đấy hôm qua vừa gửi đấy hôm qua đóng hôm qua rồi đấy qua đóng tổng cộng cho các thứ tổng cộng bốn triệu hơn bốn triệu đấy đúng rồi còn quá hạn ấy thì một tháng đấy thế còn tất toán thì đi cướp răng hả đi ăn cướp ra mà tất toán được thì chả lịch sự ở đây gì nữa chứ không lịch sự còn gì chả liên quan đéo hay đến tầm vi bi banh cả hiểu không mày hiểu vấn đề là đéo liên quan đến vi pi banh không tao thích thế đấy bọn mày phải hỏi đúng mục đích bọn đúng mục tiêu hiểu không địt mẹ bọn mày bọn mày làm vi bi banh cho bọn mày làm tín dụng khác à liên quan đéo gì đến tao như nào ờ thì làm sao thôi thế đéo có tiền đây làm gì liếm được bay hay không thế bây giờ đéo có tiền hay nào thế nào được ăn thịt tao này ờ đi kệ mẹ nó ớ thì kệ tao ừ đi kệ tao thì làm sao chả có gì mất lịch sự cả người ta biết mày là ai do tao mày tao
  PREDICTED: hôm vừa vừa hôm hôm hôm qua đấy qua qua tổng tổng các có hơn bốn đấy rồi quá đấy đấy tất tất toán đi ănt ra ra mà toán cả không ch liên liên đo đến vi hiểu hiểu hiểu hiểu vấn vấn là là ba bọn hỏi đúng hiểu đ bọn làm cho bọn bọn làm làm à như  đó có đấy v không thế tiền nào thì mẹ mẹ  thì k tao tao tao k tao tao sao mất biết ai

--- Processed 19/29 samples. ---
Sample:
  REFERENCE: à a lô ạ chị cúc đang nghe máy đúng không ạ vâng em chào chị em là chung anh vâng mình nghe được không ạ alo ạ ờ vầng vầng em trao đổi nhanh thôi em là chung anh gọi từ ngân hàng vi bi banh em gọi để báo khoản vay tín chấp và thế chấp của mình ngày mười lăm đến hạn một khoản bảy triệu sáu trăm hai mươi và một khoản bốn triệu bảy em làm tròn dư rồi mình để ý hôm đấy giúp em trước năm giờ chiều đến hạn nhá để tránh phát sinh thêm lãi phí phạt tránh ảnh hưởng đến tín dụng được không ạ à dạ vâng em gọi ra để báo hôm đấy cái hôm mười lăm ý ạ mình thanh toán cho em trước năm giờ chiều để tánh phát sinh thêm lãi phí phạt ý ạ vâng nếu thanh toán thì mình một khoản là bảy triệu sáu trăm hai mươi và một khoản là bốn triệu bảy em làm tròn dư rồi chị ạ tổng hai khoản luôn đấy thì vâng mình vẫn chuyển khoản như mọi khi đúng không chị à nộp tiền mặt tại chi nhánh vâng em làm tròn là mười hai triệu ba trăm hai mươi nghìn chị nhá chú ý tránh phát sinh thêm em làm tròn dư rồi nên mình cứ đánh thanh toán theo số này là được vầng em cảm ơn hai mươi nghìn đúng rồi ạ em làm tròn dư rồi đúng rồi ạ cảm ơn chị ờ bây giờ có theo thôi em chào chị ạ
  PREDICTED: đúng em chung mình được được không a a ạ em tra thôi em anh ng em để báo khoản t của mười đến khoảng khoảng triệu trăm trăm hai còn bảy tháng mình mình mấy trước trước giờ chiều chiều đến nh để để phát phát thêm thêm phải phải dụng em em ra để báo báo hôm ngày ngày hôm mười thanh thanh giúp trước trước để để phát phát hiện hiện ph ph thanh khoảng khoảng bảy bảy sáu sáu hai hai là triệu em em tháng rồi hai hai đấy v mình mình chuyển chuyển như l đúng đúng tại v v em hai ba hai chị xử chủ phát phát trên tư thanh thanh số là em em hai hai đúng ạ ạ lập d d rồi rồi rồi cảm chị vi em em

--- Processed 20/29 samples. ---
Sample:
  REFERENCE: nghe a lô mà không lên tiếng nè nghe nói chậm chậm chậm lại chút coi ờ ừ nhiêu ạ ờ không không chuyển khoản để tiền mặt luôn à mười hai triệu ba trăm hai mươi ngàn phải không rồi rồi rồi rồi rồi
  PREDICTED: không không tiền tiền luôn luôn rồi rồi

--- Processed 21/29 samples. ---
Sample:
  REFERENCE: alo alo ạ cho em hỏi đây có phải số điện thoại của anh hữu không ạ em chào anh em là hoàng anh liên hệ với anh từ chỗ phía ngân hàng vi bi banh ạ anh hữu cho em hỏi trước đó là anh có phải là chỗ anh em trai với anh huê phước hậu không ạ trao đổi lịch sử vào anh ạ anh trao đổi lịch sử và văn tục chửi bậy gì ở đây vậy anh ừ tôi kiếm anh trai em trai của anh ấy anh ạ còn anh trao đổi cho lịch tự đàng hoàng vào anh ạ mình có tuổi rồi trao đổi anh văng tục chửi bậy thế hả anh hả anh tôi nhìn thấy ảnh zalo của anh cũng sáng sủa mà hay là anh trao đổi anh như nào anh trao đổi như này bố mẹ của anh anh em của anh ở cạnh nghe thấy thì sao nghe thấy những lời lẽ này của anh thì sao hả anh hứ anh nghe rõ không anh nghe rõ không còn anh cứ trao đổi như thế này ý anh trao đổi như này ấy bên ngân hàng tôi vẫn sẽ phải liên hệ lại anh ạ bởi vì bên tôi chưa trao đổi được cái gì anh ạ liên hệ thì anh chỉ toàn văng tục chửi bậy nói anh có lời lẽ khiếm nhã như vậy thôi anh ạ còn anh trao đổi như này ấy bên tôi ngắt máy và tiếp tục liên hệ với anh vào cuộc gọi sau anh nhá chào anh anh về anh bảo anh em trai của anh là huê phước hậu đi trả nợ phía bên ngân hàng đi anh thì trốn nợ còn em thì khi mà nghe thấy cuộc gọi này thì văng tục thử bậy thái độ bất hợp tác anh nhá chào anh
  PREDICTED: điện anh không em anh hoàng hoàng anh liên với với từ từ từ phía phía ng ng anh anh cho chút chút là anh anh phải anh anh với anh hậu hậu không không anh anh anh lịch đây tôi kiếm anh em em emi của anh còn tra tra lịch lịch đà đà hoàng hoàng hoàng anh mình anh tuổi tuổi rồi rồi tra anh tục đây hả nhìn nhìn ảnh l anh khá mà mà hay như anh anh như bố bố anh anh nghe thì nghe nghe nghe l này này của thì thì hả anh ⁇  ⁇  ⁇  không anh không không anh tra như như ấy ấy anh tra như như bên vẫn vẫn phải lại anh bởi bên tôi chưa chưa được được gì anh liên liên thì anh toàn toàn ch nó anh n nh thôi anh anh anh anh tra này bên ng và tiếp tiếp liên với với cuộc cuộc anh chào chào anh anh anh bảo bảo anh của ở ph hậu hậu trả bên ng ng đi anh anh trốn trốn còn còn còn em khi khi nghe nghe thấy gọi tục tục tục thái độ độ độ bất bất anh anh chào chào

--- Processed 22/29 samples. ---
Sample:
  REFERENCE: alo anh tịnh đang nghe máy phải không ạ anh trọng tịnh đang nghe máy phải không ạ chào anh em là hồng manh gọi cho anh từ ngân hàng vi bi banh ạ anh có khoản vay thế chấp hai ngày nữa tới hạn là ba mươi tám triệu tám trăm để mình thanh toàn đúng hạn giúp em anh nhé vâng vâng mình anh chuyển khoản qua để thanh toán phải không anh anh chuyển khoản đúng không ạ ba mươi tám triệu tám trăm ạ ba mươi tám triệu tám trăm ạ anh thanh toán chuyển khoản phải không ạ dạ vầng thế em xác nhận là mình thanh toán đúng hạn nhá ba mươi tám triệu tám trăm chuyển khoản để tránh phát sinh phí phạt ạ
  PREDICTED: anh anh máy phải anh trọng chnh không à à chào chào em em anh anh ng anh khoản t chấp chấp hai là hai anh anh có anh hai ba ba tám tám tám đúng đúng giúp giúp anh v v anh anh khoản khoản qua để thanh thanh phải anh anh anh chuyển chuyển đúng không ba ba tám tám tám tám ba tám tám tám tám anh anh chuyển phải v v thanh thanh đúng đúng để tránh

--- Processed 23/29 samples. ---
Sample:
  REFERENCE: alo anh nghe nè ờ anh nghe anh nghe nè ờ em ờ ờ ờ ờ ờ bao nhiêu em hả ok ok em ờ ờ
  PREDICTED: nghe này bao em em k

--- Processed 24/29 samples. ---
Sample:
  REFERENCE: a lô chị hiền linh đang nghe máy phải không ạ a alo anh nhật linh đang nghe máy ạ à chào anh em là hồng anh gọi cho anh từ ngân hàng vi bi banh ạ anh có khoản vay thể chấp và hai ngày nữa tới hạn là một triệu bốn trăm năm mươi ngàn ý thì anh thanh toán đúng hạn giúp em ạ em là hồng anh gọi cho anh từ ngân hàng vi bi banh ý thì đầu năm mới thì ờ vi bi banh cũng ờ kính chúc anh chị và gia đình an khang thịnh vượng anh nhá dạ vâng thì anh có một khoản vay thấu chi thì ngày mai tới hạn luôn ạ thì là tám trăm nghìn thì mình thanh toán đúng hạn giúp em à hai khoản vay thì ờ hai ngày nữa tới hạn hai khoản vay một khoản là một triệu bốn trăm năm mươi ngàn một khoản là hai triệu tư nhá anh nhá tất toán đúng bạn luôn giúp em với cả thẻ tín dụng bốn ngày nữa tới hạn ạ là bảy trăm nghìn em làm tròn rồi ạ anh thanh toán đúng hạn giúp em bình thường mình chuyển khoản qua để thanh toán phải không anh dạ vầng dạ vầng vậy em xác nhận là mình thanh toán đúng hạn nhá chuyển khoản tránh phát sinh phí phạt anh nhá vâng cảm ơn anh em chào anh
  PREDICTED: không anh linh ạ chào em hồ anh gọi anh từng anh có chấp hai tới tới là là triệu trăm ngàn ngànn thì thì thanh đúng đúng giúp ạ là hồ hồ gọi từ ng hàng thì thì đầu thì vi vi ba chính anh anh vào đình thì có th th thì thì may luôn là trăm nghìn thì thanh đúng đúng giúp hai hai khoản  nhớ tới tới khoản khoản v một bốn năm ngàn ngàn khoản hai hai anh đúng đúng th th t bốn bốn n t hạn hạn là là là bảy bảy nghìn nghìn đúng giúp giúp bình mình mình khoản khoản hạn thanh thanh phải không v em xác là thanh thanh đúng đúng chuyển chuyển khoản tránh tránh ph ph ph anh v v cảm cảm anh anh chào

--- Processed 25/29 samples. ---
Sample:
  REFERENCE: dạ vâng ạ dạ dạ rồi dạ rồi ạ dạ em nhận rồi dạ dạ dạ rồi dạ rồi ạ dạ dạ dạ rồi
  PREDICTED: 

--- Processed 26/29 samples. ---
Sample:
  REFERENCE: alo ạ chào anh có phải anh cường đang nghe máy phải không ạ em tên là thanh từ bên ngân hàng vi bi banh ạ anh cường đang có khoản thế chấp và khoản tiêu dùng năm ngày nữa tới hạn thanh toán ạ anh đã nhận được thông báo của ngân hàng chưa anh chín triệu sáu trăm nghìn đồng ạ dạ vâng vậy anh sắp xếp thanh toán đúng hạn cho ngân hàng tránh phát sinh phí lãi phạt ảnh hưởng uy tín dụng và chuyển nhóm nợ nhá chuyển khoản hoặc thanh toán a tiền mặt cho ngân hàng vào ngày mười lăm này ạ vâng ạ vâng và em nhận nhở hả anh thanh dạ vâng vậy anh thanh em ghi nhận anh thanh toán đúng hạn trên áp vi bi banh giúp em ạ dạ em cảm ơn em chào anh ạ
  PREDICTED: a ạ anh phải anh nghe không cho anh bên ng anh anh đang khoản thấp thấp và khoản tiêu trong năm tới thanh thanh thanh ạ đã đã được thông của ở ng hà chưa chín dạ dạ anh xếp xếp đúng cho ng ng tránh phát phát sinh ph ph ảnh ảnh   t t t chuyển chuyển nh chuyển chuyển chuyển chuyển hoặc hoặc tiền tiền cho ng ng và mười này vng em anh dạ anh em anh a giúp em anh em

--- Processed 27/29 samples. ---
Sample:
  REFERENCE: ờ đúng rồi em à rồi em ờ ok ok ok nó có báo bên áp mà em
  PREDICTED: 

--- Processed 28/29 samples. ---
Sample:
  REFERENCE: alo em chào chị chị có phải là chị lan đang nghe máy đúng không chị em liên hệ chị từ bên ngân hàng vi bi banh có phải chị lan đang nghe máy đúng không ạ nhân dịp năm mới doanh kính chúc chị và gia đình an khang thịnh vượng ạ thì ờ em liên hệ với chị về khoản vay thế chấp hai ngày nữa tới hạn thanh toán ạ một khoản tiêu dùng cũng hai ngày nữa tới hạn tổng thanh toán của chị là mười lăm ờ mười sáu triệu năm trăm nghìn ạ chị đã nhận được thông báo của ngân hàng chưa ạ dạ vâng vâng tức là khoản tiêu dùng và khoản thuế thấp của chị mờ hai ngày nữa tới hạn ạ tổng là mười sáu triệu năm trăm nghìn chị đã nhận được thông báo của ngân hàng chưa chị a lô chị có nghe rõ không ạ vâng đúng rồi ạ vi pi banh ạ ngân hàng việt nam thịnh vượng ạ vào ngày mùng năm tới hạn ấy ạ thì thư vâng vậy chị thu xếp thanh toán đúng hạn cho ngân hàng ngày mười lăm này chuyển khoản hay là thanh toán tiền mặt thế chị alo dạ vâng chị thu xếp thanh toán đúng hạn cho ngân hàng chuyển khoản vào ngày mười lăm ngày mười sáu triệu năm trăm nghìn đồng chị nhá dạ em cảm ơn em chào chị ạ để tránh phát tinh phí lãi phạt và ảnh hưởng uy tín dụng chuyển nhóm nợ ấy ạ dạ em cảm ơn em chào chị ạ em xin phép ngắt máy
  PREDICTED: a có là thị đang nghe không không cho chị ng có chị la nghe nghe không chị gia đình thị em liên với với về về khoản khoản khoản thấp hai hai tới thanh thanh khoản khoản khoản tiêu tiêu cũng cũng ngày ngày tới tới thanh thanh của của mười mười mười triệu triệu trăm trăm chị đã được thông của của ng hàng dạ dạ tức tức khoản khoản dùng khoản thấp thấp của của hai hai tới tổng tổng sáu chị đã đã thông thông của của ng chưa a l chị có rõ v v đúng ng ng việt thị thị ạ ngày m m thì v v vậy xếp thanh cho cho ng chuyển chuyển hay thanh thanh tiền tiền thế dạ dạ chị chị xếp đúng đúng cho chuyển chuyển chuyển vào sáu đồng đồng nh nh dạ em ơn em em chào để phát phát l và và t n n ạ em chị em em máy em chào

--- Processed 29/29 samples. ---
Sample:
  REFERENCE: alo alo phải phải có gì không alo nghe nói lớn lớn lên chợ ồn lắm không nghe tới hai ngày nữa hả bên nào bên nào chưa chưa nghe nghe nói đi
  PREDICTED: bên bên

✅ Finished testing.
====================================================================================================
✨ Final WER for the test set: 0.6782
(Calculated on 29 samples)
====================================================================================================
