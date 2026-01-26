import soundfile as sf
import torch
import time
import threading
from qwen_tts import Qwen3TTSModel
import pynvml

# Initialize NVML for GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# GPU monitoring state
monitoring = False
gpu_stats = []

def monitor_gpu():
    """Background thread to monitor GPU utilization."""
    while monitoring:
        util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_stats.append({
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "mem_used_gb": mem_info.used / 1024**3,
            "mem_total_gb": mem_info.total / 1024**3
        })
        time.sleep(0.5)

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

total_start = time.time()

for speaker in speakers:
    print(f"\n{'='*50}")
    print(f"Generating audio for {speaker}...")

    # Reset GPU stats and start monitoring
    gpu_stats.clear()
    monitoring = True
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()

    start_time = time.time()
    wavs, sample_rate = tts.generate_custom_voice(text=text, speaker=speaker, language="english", emotion=emotions)
    elapsed = time.time() - start_time

    # Stop monitoring
    monitoring = False
    monitor_thread.join(timeout=1)

    filename = f"output_{speaker}.wav"
    sf.write(filename, wavs[0], sample_rate)

    # Calculate GPU stats
    if gpu_stats:
        avg_gpu = sum(s["gpu_util"] for s in gpu_stats) / len(gpu_stats)
        max_gpu = max(s["gpu_util"] for s in gpu_stats)
        avg_mem = sum(s["mem_used_gb"] for s in gpu_stats) / len(gpu_stats)
        print(f"  GPU Utilization: avg={avg_gpu:.1f}%, max={max_gpu:.1f}%")
        print(f"  VRAM Used: {avg_mem:.2f} GB")

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Saved to: {filename}")

total_elapsed = time.time() - total_start
print(f"\n{'='*50}")
print(f"Generated {len(speakers)} audio files in {total_elapsed:.2f}s")
print(f"Average time per speaker: {total_elapsed/len(speakers):.2f}s")

pynvml.nvmlShutdown()
