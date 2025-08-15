import kagglehub
import os
import librosa

# Download latest version
path = kagglehub.dataset_download("kynthesis/vivos-vietnamese-speech-corpus-for-asr")
path = os.path.join(path, "vivos")
# path = "datasets/vivos"
# train_audio_dir = os.path.join(path, "train", "augumented_8k_waves")
train_audio_dir = os.path.join(path, "train", "waves")
train_promts_path = os.path.join(path, "train", "prompts.txt")
# test_audio_dir = os.path.join(path, "test", "augumented_8k_waves")
test_audio_dir = os.path.join(path, "test", "waves")
test_promts_path = os.path.join(path, "test", "prompts.txt")

import json

def create_manifest(audio_dir, prompts_path, manifest_path, append=True):
    print(f"Creating manifest at {manifest_path} from {audio_dir} and {prompts_path}")
    prompts = {}
    if append:
        mode = "a"
    else:
        mode = "w"
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().partition(' ')
            if len(parts) == 3:
                audio_id, _, transcript = parts
                prompts[audio_id] = transcript
    with open(manifest_path, mode, encoding="utf-8") as out_f:
        audio_dirs = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir)]
        stats = {"max_duration": 0, "min_duration": float('inf'), "total_duration": 0, "count": 0, "avg_duration": 0}

        for audio_dirs in audio_dirs:
            audio_files = [os.path.join(audio_dirs, f) for f in os.listdir(audio_dirs) if f.endswith('.wav')]
            for audio_file in audio_files:
                audio_id = os.path.splitext(os.path.basename(audio_file))[0]
                transcript = prompts.get(audio_id)
                
                if transcript:
                    transcript = transcript.strip().lower()
                    duration = librosa.core.get_duration(path=audio_file)
                    sample_rate = librosa.get_samplerate(audio_file)
                    if sample_rate != 16000:
                        print(f"Warning: {audio_file} has sample rate {sample_rate}, expected 16000.")
                        continue
                    entry = {
                        "audio_filepath": audio_file,
                        "duration": duration,
                        "text": transcript,
                        "sample_rate": sample_rate
                    }
                    stats["max_duration"] = max(stats["max_duration"], duration)
                    stats["min_duration"] = min(stats["min_duration"], duration)
                    stats["total_duration"] += duration
                    stats["count"] += 1
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if stats["count"] > 0:
            stats["avg_duration"] = stats["total_duration"] / stats["count"]
            print(f"Stats - Max Duration: {stats['max_duration']}, Min Duration: {stats['min_duration']}, Total Duration: {stats['total_duration']}, Count: {stats['count']}, Avg Duration: {stats['avg_duration']}")        
os.makedirs("datasets/vivos", exist_ok=True)

create_manifest(train_audio_dir, train_promts_path, "datasets/vivos/train_manifest.json", append=True)
create_manifest(test_audio_dir, test_promts_path, "datasets/vivos/test_manifest.json", append=True)

print("Path to dataset files:", path)
print("Created train_manifest.json and test_manifest.json")