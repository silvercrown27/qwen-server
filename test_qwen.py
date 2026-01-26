import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained("./Qwen3-TTS-12Hz-1.7B-CustomVoice")

print("Available speakers:", tts.get_supported_speakers())

text = "Hello! This is a test of the Qwen 3 TTS system. It supports multiple languages and natural speech generation."

wavs, sample_rate = tts.generate_custom_voice(text=text, speaker="vivian", language="english")

sf.write("output.wav", wavs[0], sample_rate)
print("Audio saved to output.wav")
