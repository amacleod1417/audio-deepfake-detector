import os
import librosa
import soundfile as sf

RAW_DIR = "dataset/raw"
PROCESSED_DIR = "dataset/preprocessed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_audio(file_path, save_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    sf.write(save_path, y, sr)

if __name__ == "__main__":
    for file in os.listdir(RAW_DIR):
        if file.endswith(".wav") or file.endswith(".mp3"):
            input_path = os.path.join(RAW_DIR, file)
            output_path = os.path.join(PROCESSED_DIR, os.path.splitext(file)[0] + ".wav")
            preprocess_audio(input_path, output_path)
            print(f"Processed {file}")
