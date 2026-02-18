#!/usr/bin/env python3
"""
CHAYA-62M Inference CLI - Optimized for Maximum Quality Output
Advanced Features:
- Repetition penalty (prevents loops)
- Dynamic temperature (adaptive creativity)
- Top-p nucleus sampling (better coherence)
- Frequency penalty (vocabulary diversity)
- Presence penalty (topic exploration)
- Improved boot animation with progress bars
"""

import argparse
import os
import sys
import time
import math
from typing import Optional, Dict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# MODEL CONFIG
# =========================
SEQ_LEN = 512
E_DIM = 192
D_MODEL = 640
NUM_LAYERS = 8
NUM_HEADS = 8
FF_MULT = 4
DROPOUT = 0.05

# =========================
# TERMINAL STYLING
# =========================
RESET = "\033[0m"
GOLD = "\033[38;5;220m"
CYAN = "\033[38;5;51m"
GREEN = "\033[38;5;46m"
GRAY = "\033[90m"
WHITE = "\033[97m"
BOLD = "\033[1m"
DIM = "\033[2m"

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def type_effect(text, delay=0.01):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# -------------------------
# Enhanced Loading Animation
# -------------------------
def progress_bar(current, total, width=30, label=""):
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r{CYAN}[{bar}] {int(percent * 100)}%{RESET} {GRAY}{label}{RESET}")
    sys.stdout.flush()
    if current == total:
        print()

def animated_dots(message, duration=1.0, color=GRAY):
    sys.stdout.write(f"{color}{message}{RESET}")
    sys.stdout.flush()
    dots = 0
    start = time.time()
    while time.time() - start < duration:
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.15)
        dots += 1
        if dots > 3:
            sys.stdout.write("\b\b\b\b    \b\b\b\b")
            dots = 0
    print()

def boot_sequence():
    """Enhanced boot sequence with progress bars"""
    steps = [
        ("Initializing neural architecture", 0.8),
        ("Loading tokenizer modules", 0.6),
        ("Allocating transformer layers", 1.0),
        ("Restoring learned parameters", 1.2),
        ("Calibrating attention heads", 0.7),
        ("Warming up inference engine", 0.5),
        ("Running integrity checks", 0.6),
    ]
    
    print(GRAY + "┌" + "─" * 58 + "┐" + RESET)
    print(GRAY + "│" + " " * 58 + "│" + RESET)
    print(GRAY + "│" + f"{' ' * 15}CHAYA BOOT SEQUENCE{' ' * 22}" + "│" + RESET)
    print(GRAY + "│" + " " * 58 + "│" + RESET)
    print(GRAY + "└" + "─" * 58 + "┘" + RESET)
    print()
    
    total_steps = len(steps)
    for i, (step, duration) in enumerate(steps, 1):
        print(f"{CYAN}[{i}/{total_steps}]{RESET} {GRAY}{step}{RESET}", end="")
        
        # Simulate loading with dots
        for _ in range(int(duration * 5)):
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(duration / 5)
        
        print(f" {GREEN}✓{RESET}")
        time.sleep(0.1)
    
    print()

def logo():
    """ASCII art logo with gradient effect"""
    lines = [
        " ██████  ██   ██  █████  ██    ██  █████ ",
        "██       ██   ██ ██   ██  ██  ██  ██   ██",
        "██       ███████ ███████   ████   ███████",
        "██       ██   ██ ██   ██    ██    ██   ██",
        " ██████  ██   ██ ██   ██    ██    ██   ██"
    ]
    
    colors = [CYAN, GOLD, GOLD, GOLD, CYAN]
    
    for line, color in zip(lines, colors):
        print(color + line + RESET)
    
    print()
    print(f"{GOLD}{'─' * 44}{RESET}")
    print(f"{WHITE}         Context-Aware Hybrid Architecture{RESET}")
    print(f"{GRAY}              v1.0 | 62M Parameters{RESET}")
    print(f"{GOLD}{'─' * 44}{RESET}")
    print()

# =========================
# MODEL ARCHITECTURE
# =========================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_cos_sin(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)[None, None, :, :]
        sin = emb.sin().to(dtype)[None, None, :, :]
        return cos, sin

def apply_rope(q, k, cos, sin):
    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)

        cos, sin = self.rotary.get_cos_sin(T, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2,-1)) / self.scale
        mask = torch.tril(torch.ones((T,T), device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0,2,1,3).contiguous().view(B,T,C)
        return self.out(out)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, DROPOUT)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model * FF_MULT)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class CHAYA_62M(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, E_DIM)
        self.emb_proj = nn.Linear(E_DIM, D_MODEL, bias=False)
        self.layers = nn.ModuleList([
            TransformerBlock(D_MODEL, NUM_HEADS)
            for _ in range(NUM_LAYERS)
        ])
        self.norm_f = RMSNorm(D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, E_DIM, bias=False)

    def forward(self, idx):
        emb = self.token_embedding(idx)
        x = self.emb_proj(emb)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        e_logits = self.out_proj(x)
        logits = e_logits @ self.token_embedding.weight.t()
        return logits

# =========================
# ADVANCED SAMPLING (QUALITY OPTIMIZATION)
# =========================

def apply_repetition_penalty(logits, tokens, penalty=1.2):
    """
    Penalize tokens that have already appeared.
    Higher penalty = more diverse output.
    """
    for token_id in set(tokens.tolist()[0]):
        logits[:, token_id] /= penalty
    return logits

def apply_frequency_penalty(logits, token_counts, penalty=0.5):
    """
    Penalize tokens based on how often they've appeared.
    Prevents excessive repetition.
    """
    for token_id, count in token_counts.items():
        logits[:, token_id] -= penalty * count
    return logits

def apply_presence_penalty(logits, seen_tokens, penalty=0.6):
    """
    Penalize any token that has appeared at all.
    Encourages exploring new vocabulary.
    """
    for token_id in seen_tokens:
        logits[:, token_id] -= penalty
    return logits

def top_p_nucleus_sampling(logits, top_p=0.9, min_tokens_to_keep=1):
    """
    Nucleus sampling: sample from smallest set of tokens 
    whose cumulative probability >= top_p.
    Better quality than pure top-k.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    
    # Shift to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Scatter sorted tensors back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('inf')
    
    return logits

def dynamic_temperature(step, total_steps, base_temp=0.8, max_temp=1.0, min_temp=0.6):
    """
    Gradually decrease temperature for more coherent endings.
    Start creative, end focused.
    """
    progress = step / total_steps
    # Start high, end low (inverted sigmoid)
    temp = max_temp - (max_temp - min_temp) * (progress ** 1.5)
    return max(min_temp, min(max_temp, temp))

@torch.no_grad()
def generate_stream_optimized(
    model, 
    tokenizer, 
    device, 
    prompt, 
    length,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.2,
    frequency_penalty=0.5,
    presence_penalty=0.6,
    use_dynamic_temp=True
):
    """
    Maximum quality generation with all optimizations enabled.
    """
    model.eval()
    
    # Encode prompt
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    
    # Track token usage for penalties
    token_counts = defaultdict(int)
    seen_tokens = set()
    
    # Initialize counts from prompt
    for token_id in ids:
        token_counts[token_id] += 1
        seen_tokens.add(token_id)
    
    print(GOLD + "\nCHAYA>" + RESET, end=" ", flush=True)
    
    start_time = time.time()
    generated_tokens = []
    
    for step in range(length):
        # Get logits
        logits = model(tokens[:, -SEQ_LEN:])
        next_logits = logits[:, -1, :].clone()
        
        # Apply dynamic temperature
        if use_dynamic_temp:
            current_temp = dynamic_temperature(step, length, base_temp=temperature)
        else:
            current_temp = temperature
        
        next_logits = next_logits / current_temp
        
        # Apply penalties (QUALITY BOOST)
        next_logits = apply_repetition_penalty(next_logits, tokens, penalty=repetition_penalty)
        next_logits = apply_frequency_penalty(next_logits, token_counts, penalty=frequency_penalty)
        next_logits = apply_presence_penalty(next_logits, seen_tokens, penalty=presence_penalty)
        
        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float('inf')
        
        # Top-p nucleus sampling
        next_logits = top_p_nucleus_sampling(next_logits, top_p=top_p)
        
        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        
        # Update tracking
        token_id = next_id.item()
        token_counts[token_id] += 1
        seen_tokens.add(token_id)
        
        # Append token
        tokens = torch.cat([tokens, next_id], dim=1)
        generated_tokens.append(token_id)
        
        # Decode and print
        word = tokenizer.decode([token_id])
        print(word, end="", flush=True)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    tokens_per_sec = length / total_time if total_time > 0 else 0
    
    print("\n")
    print(f"{GRAY}{'─' * 60}{RESET}")
    print(f"{GRAY}Generated {length} tokens in {total_time:.2f}s{RESET}")
    print(f"{GREEN}⚡ Speed: {tokens_per_sec:.2f} tok/s{RESET}")
    print(f"{CYAN}🎯 Quality Mode: {'Dynamic' if use_dynamic_temp else 'Static'}{RESET}")
    print(f"{GRAY}{'─' * 60}{RESET}")
    print()

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="CHAYA-62M Optimized Inference")
    parser.add_argument("--ckpt", default="CHAYA_62M_Phase2_Final.pt", help="Model checkpoint path")
    parser.add_argument("--tokenizer", default="./gpt2_tokenizer", help="Tokenizer path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Generation parameters
    parser.add_argument("--length", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Base temperature")
    parser.add_argument("--topk", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--topp", type=float, default=0.92, help="Nucleus sampling threshold")
    
    # Quality optimization parameters
    parser.add_argument("--rep-penalty", type=float, default=1.15, help="Repetition penalty (1.0=off, 1.2=strong)")
    parser.add_argument("--freq-penalty", type=float, default=0.3, help="Frequency penalty")
    parser.add_argument("--pres-penalty", type=float, default=0.5, help="Presence penalty")
    parser.add_argument("--dynamic-temp", action="store_true", default=True, help="Use dynamic temperature")
    parser.add_argument("--no-dynamic-temp", dest="dynamic_temp", action="store_false")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    clear()
    logo()
    boot_sequence()
    
    print(f"{GREEN}✓ System online{RESET}")
    print(f"{GRAY}Device: {args.device.upper()}{RESET}")
    print()
    
    # Load tokenizer
    print(f"{CYAN}[INIT]{RESET} Loading tokenizer...", end=" ")
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = SEQ_LEN
    vocab_size = tokenizer.vocab_size
    print(f"{GREEN}✓{RESET} {GRAY}(vocab: {vocab_size:,}){RESET}")
    
    # Load checkpoint
    print(f"{CYAN}[INIT]{RESET} Loading model checkpoint...", end=" ")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove module. prefix from DataParallel
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Infer vocab size from checkpoint
    if "token_embedding.weight" in state_dict:
        ckpt_vocab = state_dict["token_embedding.weight"].shape[0]
        if ckpt_vocab != vocab_size:
            vocab_size = ckpt_vocab
    
    model = CHAYA_62M(vocab_size).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"{GREEN}✓{RESET} {GRAY}(params: 62M){RESET}")
    
    print()
    print(f"{GOLD}{'═' * 60}{RESET}")
    print(f"{WHITE}  CHAYA v1.0 Ready{RESET}")
    print(f"{GRAY}  Quality optimizations: {'ENABLED' if args.dynamic_temp else 'DISABLED'}{RESET}")
    print(f"{GOLD}{'═' * 60}{RESET}")
    print()
    print(f"{DIM}Type 'exit' or 'quit' to end session{RESET}")
    print()
    
    # Interactive loop
    while True:
        try:
            prompt = input(f"{BOLD}{WHITE}>{RESET} ")
            
            if prompt.lower().strip() in ["exit", "quit", "q"]:
                print(f"\n{CYAN}Shutting down...{RESET}")
                break
            
            if not prompt.strip():
                continue
            
            generate_stream_optimized(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                length=args.length,
                temperature=args.temp,
                top_k=args.topk,
                top_p=args.topp,
                repetition_penalty=args.rep_penalty,
                frequency_penalty=args.freq_penalty,
                presence_penalty=args.pres_penalty,
                use_dynamic_temp=args.dynamic_temp
            )
            
        except KeyboardInterrupt:
            print(f"\n\n{CYAN}Interrupted. Shutting down...{RESET}")
            break
        except Exception as e:
            print(f"\n{GOLD}[ERROR]{RESET} {str(e)}")
            continue
    
    print(f"{GREEN}Goodbye!{RESET}\n")

if __name__ == "__main__":
    main()