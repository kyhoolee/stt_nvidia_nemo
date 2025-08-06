import torchaudio
import soxr
import os
import torch
import shutil
import kagglehub


def resample_to_8k(waveform, sample_rate=16000):
    resampler = torchaudio.transforms.Resample(sample_rate, new_freq=8000)
    waveform_8k = resampler(waveform)
    return waveform_8k

def upsample_to_16k(audio):
    return soxr.resample(audio, 8000, 16000, quality='VHQ')

def process_audio(input_wav_path):
    waveform, sample_rate = torchaudio.load(input_wav_path)
    waveform_8k = resample_to_8k(waveform, sample_rate)
    waveform_16k = upsample_to_16k(waveform_8k.numpy()[0])
    return torch.tensor(waveform_16k).unsqueeze(0)


def process_dataset(input_dir, output_dir):
    wave_dir = os.path.join(input_dir, "waves")
    prompt_file = os.path.join(input_dir, "prompts.txt")

    # Create output directory for waves
    output_wave_dir = os.path.join(output_dir, "augumented_8k_waves")
    os.makedirs(output_wave_dir, exist_ok=True)
    
    # Copy prompts.txt to output directory
    output_prompt_file = os.path.join(output_dir, "prompts.txt")
    import shutil
    shutil.copy2(prompt_file, output_prompt_file)
    print(f"Copied prompts file to {output_prompt_file}")
    
    input_dirs = [os.path.join(wave_dir, d) for d in os.listdir(wave_dir) if os.path.isdir(os.path.join(wave_dir, d))]
    
    for dir_path in input_dirs:
        speaker_dir = os.path.basename(dir_path)
        out_speaker_dir = os.path.join(output_wave_dir, speaker_dir)
        os.makedirs(out_speaker_dir, exist_ok=True)
        
        for filename in os.listdir(dir_path):
            if filename.endswith('.wav'):
                input_wav_path = os.path.join(dir_path, filename)
                output_wav_path = os.path.join(out_speaker_dir, filename)
                processed_audio = process_audio(input_wav_path)
                torchaudio.save(output_wav_path, processed_audio, 16000)
                print(f"Processed {input_wav_path} and saved to {output_wav_path}")

if __name__ == "__main__":
    # Download the dataset
    data_path = kagglehub.dataset_download("kynthesis/vivos-vietnamese-speech-corpus-for-asr")
    # Define paths
    input_dir = os.path.join(data_path, "vivos/test")
    output_dir = os.path.join("datasets", "vivos/test")
    
    # Process the dataset
    process_dataset(input_dir, output_dir)
    
    print(f"Processed audio files saved to {output_dir}")