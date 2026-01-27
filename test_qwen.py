import soundfile as sf
import torch
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    dtype=torch.bfloat16,  # bfloat16 often faster than float16
    device_map="cuda",
    attn_implementation="flash_attention_2",  # Requires flash-attn package
)

# Enable inference optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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

Let me give you a more detailed example. Consider a roller coaster at an amusement park.
At the very top of the first hill, the car has maximum potential energy and almost zero kinetic energy.
As it races down, that potential energy converts to kinetic energy, making the car go faster and faster.
At the bottom of the hill, kinetic energy is at its maximum.

Then as the car climbs the next hill, the process reverses. Kinetic energy transforms back into potential energy.
The car slows down as it rises. Some energy is lost to friction and air resistance, becoming heat.
But if you add up all the forms of energy at any point, the total remains constant.

This same principle governs how power plants generate electricity. In a coal plant, chemical energy
stored in coal transforms into heat energy when burned. That heat boils water into steam.
The steam's thermal energy becomes mechanical energy as it spins turbines.
Finally, the turbines convert mechanical energy into electrical energy that powers our homes.

Even in your own body, conservation of energy is at work every single moment.
The food you eat contains chemical energy stored in molecular bonds.
Your digestive system breaks down these molecules, releasing that energy.
Your cells use this energy to power movement, maintain body temperature, and keep your brain thinking.
Nothing is created or destroyed. Everything is transformed.
"""

emotions = "[passionate][clear][enthusiastic]"

speakers = tts.get_supported_speakers()

# Number of parallel workers - try 1 first to see baseline, then increase
# Running multiple in parallel may cause GPU contention
MAX_WORKERS = 1

@torch.inference_mode()
def generate_for_speaker(speaker):
    """Generate audio for a single speaker."""
    start_time = time.time()
    wavs, sample_rate = tts.generate_custom_voice(text=text, speaker=speaker, language="english", emotion=emotions)
    elapsed = time.time() - start_time

    filename = f"output_{speaker}.wav"
    sf.write(filename, wavs[0], sample_rate)

    return speaker, elapsed, filename

total_start = time.time()

# Start GPU monitoring for the entire batch
gpu_stats.clear()
monitoring = True
monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
monitor_thread.start()

print(f"Generating audio for {len(speakers)} speakers with {MAX_WORKERS} parallel workers...")
print(f"{'='*50}")

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(generate_for_speaker, speaker): speaker for speaker in speakers}

    for future in as_completed(futures):
        speaker, elapsed, filename = future.result()
        print(f"  {speaker}: {elapsed:.2f}s -> {filename}")
        results.append((speaker, elapsed))

# Stop monitoring
monitoring = False
monitor_thread.join(timeout=1)

total_elapsed = time.time() - total_start

# Print summary
print(f"\n{'='*50}")
if gpu_stats:
    avg_gpu = sum(s["gpu_util"] for s in gpu_stats) / len(gpu_stats)
    max_gpu = max(s["gpu_util"] for s in gpu_stats)
    max_mem = max(s["mem_used_gb"] for s in gpu_stats)
    print(f"GPU Utilization: avg={avg_gpu:.1f}%, max={max_gpu:.1f}%")
    print(f"Peak VRAM Used: {max_mem:.2f} GB")

print(f"\nGenerated {len(speakers)} audio files in {total_elapsed:.2f}s (parallel)")
print(f"Sum of individual times: {sum(e for _, e in results):.2f}s")
print(f"Speedup: {sum(e for _, e in results) / total_elapsed:.2f}x")

pynvml.nvmlShutdown()
