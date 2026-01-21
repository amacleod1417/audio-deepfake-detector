import os
import random
import time
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=api_key)

OUTPUT_DIR = "dataset/raw/fake"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_FILE = "dataset/text/sentences.txt"
with open(PROMPT_FILE, "r") as f:
    sentence_pool = [line.strip() for line in f if line.strip()]

NUM_CLIPS = 200
if len(sentence_pool) < NUM_CLIPS:
    raise ValueError(f"Not enough sentences in {PROMPT_FILE}, found only {len(sentence_pool)}")

voices_response = client.voices.get_all()
voices = voices_response.voices  # List of Voice objects
voice_dict = {v.name: v for v in voices}
available_voice_names = list(voice_dict.keys())
print("Available voices:", available_voice_names)

random.shuffle(sentence_pool)

for i in range(NUM_CLIPS):
    text = sentence_pool[i]
    voice_name = random.choice(available_voice_names)
    voice_id = voice_dict[voice_name].voice_id

    print(f"[{i+1}/{NUM_CLIPS}] Synthesizing with voice '{voice_name}': {text}")

    audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_monolingual_v1",
        )
    filename = f"fake_eleven_{i:04d}_{voice_name}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
            for chunk in audio:
                f.write(chunk)
    time.sleep(1)
   