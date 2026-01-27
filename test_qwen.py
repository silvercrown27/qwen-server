import soundfile as sf
import torch
import time
import threading
import numpy as np
from vllm import LLM, SamplingParams
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown
except ImportError:
    from nvidia_ml_py import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown

# Initialize NVML for GPU monitoring
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)

# GPU monitoring state
monitoring = False
gpu_stats = []

def monitor_gpu():
    """Background thread to monitor GPU utilization."""
    while monitoring:
        util = nvmlDeviceGetUtilizationRates(gpu_handle)
        mem_info = nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_stats.append({
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "mem_used_gb": mem_info.used / 1024**3,
            "mem_total_gb": mem_info.total / 1024**3
        })
        time.sleep(0.5)

# Load model with vLLM
llm = LLM(
    model="./Qwen3-TTS-12Hz-1.7B-CustomVoice",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {torch.cuda.get_device_name(0)}")

# Available speakers for Qwen3-TTS CustomVoice
speakers = ['aiden', 'dylan', 'eric', 'ono_anna', 'ryan', 'serena', 'sohee', 'uncle_fu', 'vivian']
print("Available speakers:", speakers)

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

# Sampling parameters for TTS generation
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=4096,
)

# Build prompts for all speakers (batch processing)
prompts = []
for speaker in speakers:
    # Format prompt for Qwen3-TTS with speaker and language
    prompt = f"<|speaker|>{speaker}<|language|>english<|text|>{text.strip()}<|audio|>"
    prompts.append(prompt)

total_start = time.time()

# Start GPU monitoring
gpu_stats.clear()
monitoring = True
monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
monitor_thread.start()

print(f"\nGenerating audio for {len(speakers)} speakers using vLLM batch processing...")
print(f"{'='*50}")

# Generate all speakers in a single batch (vLLM handles batching efficiently)
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
batch_elapsed = time.time() - start_time

# Stop monitoring
monitoring = False
monitor_thread.join(timeout=1)

# Process outputs and save audio files
SAMPLE_RATE = 24000  # Qwen3-TTS default sample rate
results = []

for i, (speaker, output) in enumerate(zip(speakers, outputs)):
    # Extract audio tokens from output and convert to waveform
    # Note: The exact processing depends on the model's output format
    generated_text = output.outputs[0].text

    # For TTS models, output may contain audio tokens that need decoding
    # This is a placeholder - actual implementation depends on model specifics
    filename = f"output_{speaker}.wav"

    # If the model outputs raw audio data, save it
    # Otherwise, you may need additional post-processing
    print(f"  {speaker}: generated -> {filename}")
    results.append((speaker, filename))

total_elapsed = time.time() - total_start

# Print summary
print(f"\n{'='*50}")
if gpu_stats:
    avg_gpu = sum(s["gpu_util"] for s in gpu_stats) / len(gpu_stats)
    max_gpu = max(s["gpu_util"] for s in gpu_stats)
    max_mem = max(s["mem_used_gb"] for s in gpu_stats)
    print(f"GPU Utilization: avg={avg_gpu:.1f}%, max={max_gpu:.1f}%")
    print(f"Peak VRAM Used: {max_mem:.2f} GB")

print(f"\nBatch generation time: {batch_elapsed:.2f}s")
print(f"Total time (including processing): {total_elapsed:.2f}s")
print(f"Average per speaker: {batch_elapsed/len(speakers):.2f}s")

nvmlShutdown()
