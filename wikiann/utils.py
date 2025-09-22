import os
import psutil
import time
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


def measure_peak_memory(model, tokenizer, max_seq_length=128, device="cuda" if torch.cuda.is_available() else "cpu", repeat=100):
    """
    Measure peak memory usage during inference. Works for GPU and CPU.
    """
    model.eval()
    model.to(device)

    sample = tokenizer("This is a test", "Ceci est un test", return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length)
    sample = {k: v.to(device) for k, v in sample.items()}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    else:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss

    with torch.no_grad():
        start_time = time.time()
        for _ in range(repeat):
            _ = model(**sample)
        end_time = time.time()

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    else:
        mem_after = process.memory_info().rss
        peak_memory = (mem_after - mem_before) / (1024 ** 3)  # GB

    avg_time = (end_time - start_time) / repeat
    return round(peak_memory, 3), round(avg_time, 4)

    
# Wrapper to make HuggingFace model compatible with fvcore
class WrappedHFModel(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def estimate_flops_fvcore(model, tokenizer, max_seq_length=128, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    model.to(device)

    wrapped_model = WrappedHFModel(model)

    # Create dummy inputs
    input_ids = torch.ones((1, max_seq_length), dtype=torch.long).to(device)
    attention_mask = torch.ones((1, max_seq_length), dtype=torch.long).to(device)

    try:
        flops = FlopCountAnalysis(wrapped_model, (input_ids, attention_mask))
        total_flops = flops.total() / 1e9  # Convert to GFLOPs
        return round(total_flops, 2)
    except Exception as e:
        print(f"[FLOPs ERROR] {e}")
        return None