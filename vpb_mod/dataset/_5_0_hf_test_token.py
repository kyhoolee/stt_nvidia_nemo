from datasets import load_dataset

# Nếu đã login thì không cần truyền token, nếu export HF_TOKEN thì cũng tự nhận
ds = load_dataset("NhutP/VietSpeech", split="train", streaming=True)

print(next(iter(ds)))
