import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# Load model with GPU optimization
tts = Qwen3TTSModel.from_pretrained(
    "./Qwen3-TTS-12Hz-1.7B-CustomVoice",
    dtype=torch.float16,
    device_map="cuda"
)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {torch.cuda.get_device_name(0)}")

print("Available speakers:", tts.get_supported_speakers())

text = """
The law of conservation of energy is one of the most fundamental principles in all of physics.
It states that energy cannot be created or destroyed, only transformed from one form to another.

Think about it this way: when you drop a ball from a height, the potential energy it had at the top
converts into kinetic energy as it falls. When it hits the ground, that energy transforms into sound
and heat. But the total amount of energy? It stays exactly the same!

This principle applies everywhere in our universe. From the chemical energy in your breakfast
becoming the mechanical energy that moves your muscles, to the nuclear reactions in the sun
that eventually reach us as light and warmth. Energy is always conserved.

Understanding this law helps us make sense of everything from simple machines to complex ecosystems.
It's truly one of nature's most beautiful and reliable rules.
"""

emotions = "[passionate][clear][enthusiastic]"

speakers = tts.get_supported_speakers()

for speaker in speakers:
    print(f"Generating audio for {speaker}...")
    wavs, sample_rate = tts.generate_custom_voice(text=text, speaker=speaker, language="english", emotion=emotions)
    filename = f"output_{speaker}.wav"
    sf.write(filename, wavs[0], sample_rate)
    print(f"Audio saved to {filename}")

print(f"\nGenerated {len(speakers)} audio files.")
