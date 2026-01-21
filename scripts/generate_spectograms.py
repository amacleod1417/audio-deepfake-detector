import os
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

real_audio_dir = "dataset/raw/real"
fake_audio_dir = "dataset/raw/fake"
real_spec_dir = "dataset/spectrograms/train/real"
fake_spec_dir = "dataset/spectrograms/train/fake"

# supported audio types
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

def convert_audio_to_spectrogram(input_path, output_path):
    y, sr = librosa.load(input_path, sr=16000)

    # fixed length of 4 seconds
    target_len = 4 * sr
    if len(y) < target_len:
        # Loop audio if too short
        n_repeats = int(np.ceil(target_len / len(y)))
        y = np.tile(y, n_repeats)[:target_len]
    elif len(y) > target_len:
        # Random crop if too long 
        start = np.random.randint(0, len(y) - target_len)
        y = y[start : start + target_len]

    # add white noise to all samples
    noise_amp = 0.02 * np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 0
    y = y + noise_amp * np.random.normal(size=len(y))

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, cmap="magma")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_audio_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(AUDIO_EXTENSIONS):
            file_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")

            try:
                # If not WAV, convert to temp wav
                if not filename.lower().endswith(".wav"):
                    audio = AudioSegment.from_file(file_path)
                    temp_path = "temp.wav"
                    audio.export(temp_path, format="wav")
                    spectro_input = temp_path
                else:
                    spectro_input = file_path

                # Convert to spectrogram
                convert_audio_to_spectrogram(spectro_input, output_path)
                print(f"✓ {filename} → spectrogram")

                # Remove temp file if used
                if spectro_input == "temp.wav":
                    os.remove("temp.wav")
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
                if os.path.exists("temp.wav"):
                    os.remove("temp.wav")


# Clear old data 
if os.path.exists("dataset/spectrograms"):
    shutil.rmtree("dataset/spectrograms")

process_audio_folder(fake_audio_dir, fake_spec_dir)
process_audio_folder(real_audio_dir, real_spec_dir)

print(" All spectrograms generated.")
