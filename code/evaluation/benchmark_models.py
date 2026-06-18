"""
benchmark_models.py
Compute FLOPs, latency, memory, and parameter counts for all models.
No dataset required — uses dummy inputs.
"""

# --- path bootstrap (code reorganized into core/ models/ configs/) ---
import os as _os, sys as _sys
_CODE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "models", "configs"):
    _p = _os.path.join(_CODE_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import time
import torch
import torch.nn as nn
import numpy as np

# ============================================================
# HELPERS
# ============================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def count_flops(model, input_tensor, device=None):
    """Count FLOPs using PyTorch's built-in flop counter."""
    from torch.utils.flop_counter import FlopCounterMode
    if device is not None:
        model = model.to(device)
        input_tensor = input_tensor.to(device)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(input_tensor)
    if device is not None:
        model = model.cpu()
    return flop_counter.get_total_flops()

def measure_latency(model, input_tensor, device, warmup=10, runs=50):
    """Measure inference latency."""
    model = model.to(device)
    x = input_tensor.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return np.median(times) * 1000  # ms

def measure_gpu_memory(model, input_tensor, device):
    """Measure peak GPU memory during inference."""
    if device.type != 'cuda':
        return 0
    model = model.to(device)
    x = input_tensor.to(device)
    model.eval()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        _ = model(x)

    peak = torch.cuda.max_memory_allocated(device) / 1e6  # MB
    return peak

# ============================================================
# MODELS
# ============================================================
def get_models():
    models = {}

    # MobileNet Student
    from student_mobilenet import MobileNetStudent
    models['MobileNetV2 Student'] = MobileNetStudent(input_nc=4)

    # TinyCNN Student
    from student_tinycnn import TinyCNNStudent
    models['TinyCNN Student'] = TinyCNNStudent(input_nc=4)

    # DLoc Baseline (encoder + decoder only, test-time config)
    try:
        from Generators import ResnetEncoder, ResnetDecoder

        encoder = ResnetEncoder(input_nc=4, output_nc=256, ngf=64, n_blocks=6)
        decoder = ResnetDecoder(input_nc=256, output_nc=1, ngf=64, n_blocks=9, encoder_blocks=6)

        class BaselineTestTime(nn.Module):
            def __init__(self, enc, dec):
                super().__init__()
                self.encoder = enc
                self.decoder = dec
            def forward(self, x):
                return torch.sigmoid(self.decoder(self.encoder(x)))

        models['DLoc Baseline (test-time)'] = BaselineTestTime(encoder, decoder)
    except Exception as e:
        print(f'Could not load baseline: {e}')

    # Mamba (try, may fail without mamba-ssm)
    try:
        from mamba_model import MambaDLocNet
        models['Mamba SSM'] = MambaDLocNet(input_nc=4)
    except Exception as e:
        print(f'Could not load Mamba: {e}')

    return models

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda:0') if torch.cuda.is_available() else None

    dummy = torch.randn(1, 4, 161, 361)

    models = get_models()

    print('=' * 90)
    print(f'{"Model":<28} {"Params":>10} {"FLOPs":>12} {"CPU (ms)":>10} {"GPU (ms)":>10} {"GPU Mem":>10}')
    print(f'{"":28} {"":>10} {"":>12} {"(batch=1)":>10} {"(batch=1)":>10} {"(MB)":>10}')
    print('=' * 90)

    results = []

    for name, model in models.items():
        model.eval()

        # Params
        n_params = count_params(model)

        # FLOPs (try CPU first, fall back to GPU for CUDA-only models)
        try:
            flops = count_flops(model, dummy)
        except Exception:
            try:
                if device_gpu:
                    flops = count_flops(model, dummy, device=device_gpu)
                else:
                    flops = 0
            except Exception as e:
                print(f'  FLOPs failed for {name}: {e}')
                flops = 0

        # CPU latency
        try:
            cpu_ms = measure_latency(model, dummy, device_cpu, warmup=3, runs=10)
        except Exception as e:
            print(f'  CPU latency failed for {name} (CUDA-only model)')
            cpu_ms = 0

        # GPU latency + memory
        gpu_ms = 0
        gpu_mem = 0
        if device_gpu:
            try:
                gpu_ms = measure_latency(model, dummy, device_gpu, warmup=10, runs=50)
                gpu_mem = measure_gpu_memory(model, dummy, device_gpu)
            except Exception as e:
                print(f'  GPU failed for {name}: {e}')

        # Format
        param_str = f'{n_params/1e6:.2f}M' if n_params > 1e6 else f'{n_params/1e3:.1f}K'
        flop_str = f'{flops/1e9:.2f}G' if flops > 1e9 else f'{flops/1e6:.1f}M' if flops > 0 else 'N/A'
        cpu_str = f'{cpu_ms:.1f}' if cpu_ms > 0 else 'N/A'
        gpu_str = f'{gpu_ms:.1f}' if gpu_ms > 0 else 'N/A'
        mem_str = f'{gpu_mem:.1f}' if gpu_mem > 0 else 'N/A'

        print(f'{name:<28} {param_str:>10} {flop_str:>12} {cpu_str:>10} {gpu_str:>10} {mem_str:>10}')

        results.append({
            'name': name,
            'params': n_params,
            'flops': flops,
            'cpu_ms': cpu_ms,
            'gpu_ms': gpu_ms,
            'gpu_mem_mb': gpu_mem,
        })

        # Clean up GPU
        if device_gpu:
            model.cpu()
            torch.cuda.empty_cache()

    print('=' * 90)

    # Weight sizes
    print('\nWeight file sizes:')
    import os
    weight_files = {
        'DLoc Baseline': ['weights/baseline_encoder_best.pth', 'weights/baseline_decoder_best.pth', 'weights/baseline_offset_decoder_best.pth'],
        'MobileNetV2': ['weights/mobilenet_standalone_best.pth'],
        'TinyCNN': ['weights/tinycnn_standalone_best.pth'],
        'Mamba': ['weights/mamba_best.pth'],
    }
    for label, files in weight_files.items():
        total = 0
        for f in files:
            if os.path.exists(f):
                total += os.path.getsize(f)
        if total > 0:
            size_str = f'{total/1e6:.1f} MB' if total > 1e6 else f'{total/1e3:.0f} KB'
            print(f'  {label}: {size_str}')
        else:
            print(f'  {label}: not found')

    # Summary
    print('\n--- LaTeX Table ---')
    print(r'\begin{tabular}{lrrrrrr}')
    print(r'\toprule')
    print(r'\textbf{Model} & \textbf{Params} & \textbf{FLOPs} & \textbf{CPU (ms)} & \textbf{GPU (ms)} & \textbf{GPU Mem (MB)} & \textbf{Weight Size} \\')
    print(r'\midrule')
    for r in results:
        param_str = f'{r["params"]/1e6:.2f}M' if r["params"] > 1e6 else f'{r["params"]/1e3:.0f}K'
        flop_str = f'{r["flops"]/1e9:.2f}G' if r["flops"] > 1e9 else f'{r["flops"]/1e6:.0f}M' if r["flops"] > 0 else 'N/A'
        cpu_str = f'{r["cpu_ms"]:.1f}' if r["cpu_ms"] > 0 else 'N/A'
        gpu_str = f'{r["gpu_ms"]:.1f}' if r["gpu_ms"] > 0 else 'N/A'
        mem_str = f'{r["gpu_mem_mb"]:.0f}' if r["gpu_mem_mb"] > 0 else 'N/A'
        print(f'{r["name"]} & {param_str} & {flop_str} & {cpu_str} & {gpu_str} & {mem_str} & --- \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
