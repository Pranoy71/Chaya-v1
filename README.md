# Chaya-v1

**Context-Aware Hybrid Autoregressive Transformer Architecture**

A `62-million` parameter transformer language model built entirely from scratch, implementing modern architectural components including Rotary Positional Embeddings (RoPE), Root Mean Square Normalization (RMSNorm), and SwiGLU activation functions. Chaya-v1 is designed for efficient inference on consumer hardware while maintaining competitive performance for its parameter class.

---

## Overview

Chaya-v1 represents a foundational language model trained through a two-phase curriculum: initial pre-training on WikiText-103 followed by continued training on SWIFT-700, a curated 700-million-token dataset spanning multiple domains. The model architecture prioritizes inference efficiency and context awareness, making it suitable for deployment on resource-constrained environments.

**Key Characteristics:**
- 62.3 million trainable parameters
- 512-token context window
- GPT-2 tokenizer (50,257 vocabulary)
- Trained on 1.378 billion tokens total
- Inference speed: ~6 tokens/second on AMD Ryzen 5 3500U (CPU)

---

## Architecture

Chaya-v1 implements a decoder-only transformer architecture with the following specifications:

### Model Configuration

| Component | Specification |
|-----------|---------------|
| **Embedding Dimension** | 192 |
| **Model Dimension** | 640 |
| **Number of Layers** | 8 |
| **Attention Heads** | 8 |
| **Head Dimension** | 80 |
| **FFN Hidden Dimension** | 2,560 (4x multiplier) |
| **Sequence Length** | 512 |
| **Total Parameters** | 62,334,784 |

### Architectural Components

The model incorporates several modern architectural improvements over traditional transformers:

**Positional Encoding:**
- Rotary Positional Embeddings (RoPE) for improved length generalization
- Base frequency: 10,000

**Normalization:**
- RMSNorm (Root Mean Square Normalization) applied pre-attention and pre-FFN
- Eliminates mean centering for improved efficiency

**Activation Function:**
- SwiGLU (Swish-Gated Linear Unit) in feed-forward networks
- Provides gating mechanism with SiLU non-linearity

**Attention Mechanism:**
- Multi-head self-attention with causal masking
- Scaled dot-product attention with dropout regularization

**Embedding Strategy:**
- Factorized embeddings: tokens → 192-dim → projected to 640-dim
- Weight tying between embedding and output layers
- Reduces parameter count while maintaining expressiveness

---

## Training Methodology

### Phase 1: Foundation Training

**Dataset:** WikiText-103 (raw)
- Tokens: 113 million per epoch
- Epochs: 6
- Total tokens: 678 million
- Validation perplexity: 37.87

**Objective:** Establish foundational language understanding and syntactic patterns.

### Phase 2: Domain Diversification

**Dataset:** SWIFT-700
- Tokens: 700 million
- Epochs: 1
- Composition: Multi-domain (web, narrative, educational, scientific, encyclopedic, instructional)
- Validation perplexity: 37.89
- Validation accuracy: 39.78%

**Objective:** Expand knowledge coverage and reduce domain-specific biases.

**Dataset Details:**
SWIFT-700 is available at:
- GitHub: [Pranoy71/Swift-700](https://github.com/Pranoy71/Swift-700)
- Kaggle: [swift-700m](https://www.kaggle.com/datasets/pranoy72/swift-700m)

### Training Configuration

```python
Optimizer: AdamW
Learning Rate: 1.9e-4 (Phase 1), 8e-5 (Phase 2)
Weight Decay: 0.01
Gradient Clipping: 1.0
Batch Size: 40
Warmup Steps: 2,000
Scheduler: Linear warmup with linear decay
Mixed Precision: FP16 with gradient scaling
```

**Hardware:** Kaggle 2× T4 GPUs
**Training Time:** 16-17 hours (combined phases)

---

## Repository Structure

```
Chaya-v1/
├── gpt2_tokenizer/              # GPT-2 tokenizer files
├── CHAYA_62M_Phase1_Final.pt    # Phase 1 checkpoint
├── CHAYA_62M_Phase2_Final.pt    # Final trained model
├── CHAYA-62M_Phase1_WikiText103.ipynb  # Phase 1 training notebook
├── CHAYA-62M-2ND-PHASE.ipynb    # Phase 2 training notebook
├── chayav1.py                   # Inference script
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU inference)

### Setup

```bash
# Clone repository
git clone https://github.com/Pranoy71/Chaya-v1.git
cd Chaya-v1

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Inference

```bash
python chayav1.py \
    --ckpt CHAYA_62M_Phase2_Final.pt \
    --tokenizer ./gpt2_tokenizer \
    --device cpu \
    --length 150 \
    --temp 0.8 \
    --topk 40 \
    --topp 0.9
```

**Parameters:**
- `--ckpt`: Path to model checkpoint
- `--tokenizer`: Path to tokenizer directory
- `--device`: Inference device (cpu/cuda)
- `--length`: Maximum tokens to generate
- `--temp`: Sampling temperature (default: 0.8)
- `--topk`: Top-k sampling parameter (default: 40)
- `--topp`: Nucleus sampling threshold (default: 0.9)

### Programmatic Usage

```python
import torch
from transformers import GPT2TokenizerFast

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("./gpt2_tokenizer")

# Load model
checkpoint = torch.load("CHAYA_62M_Phase2_Final.pt", map_location="cpu")
state_dict = checkpoint["model_state_dict"]

# Initialize and load weights
from chayav1 import CHAYA_62M
model = CHAYA_62M(vocab_size=50257)
model.load_state_dict(state_dict)
model.eval()

# Generate
with torch.no_grad():
    input_ids = tokenizer.encode("Your prompt here", return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    text = tokenizer.decode(output[0])
```

---

## Performance Characteristics

### Computational Efficiency

| Metric | Value |
|--------|-------|
| **Forward Pass (256 seq)** | ~250ms (CPU, single sample) |
| **Memory Footprint** | ~250MB (FP32) |
| **Inference Speed** | ~6 tokens/sec (Ryzen 5 3500U) |
| **Training Throughput** | ~2.2 batches/sec (2× T4, batch=40) |

### Model Quality

**Validation Metrics (Phase 2):**
- Perplexity: 37.89
- Token Accuracy: 39.78% (Domain-expanded validation set)

**Observed Capabilities:**
- Coherent text generation up to 100-150 tokens
- Basic factual recall from training data
- Narrative structure in creative writing tasks
- Context tracking within sequence window

**Known Limitations:**
- Factual hallucination in knowledge-intensive queries
- Topic drift beyond 200 tokens
- Limited reasoning capabilities (parameter constraint)
- Occasional repetition in extended generation
- Not fine tuned on QA tasks

---

## Technical Implementation Details

### Custom Components

All core components were implemented from scratch without relying on pretrained weights or high-level abstractions:

**RMSNorm Implementation:**
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)
```

**Rotary Positional Embeddings:**
- Precomputed frequency basis up to max sequence length
- Applied via rotation in complex plane representation
- Enables extrapolation to longer sequences at inference

**SwiGLU Activation:**
- Three-weight gated linear unit with SiLU activation
- Empirically superior to standard FFN with GELU/ReLU
- Parameter efficient compared to traditional gating mechanisms

---

## Reproducibility

### Training Notebooks

Two Jupyter notebooks document the complete training process:

1. **CHAYA-62M_Phase1_WikiText103.ipynb**
   - WikiText-103 data loading and preprocessing
   - Model instantiation and training loop
   - Validation metrics and checkpointing

2. **CHAYA-62M-2ND-PHASE.ipynb**
   - SWIFT-700 dataset integration
   - Continued training from Phase 1 checkpoint
   - Multi-GPU training with DataParallel
   - Validation at 25% intervals

Both notebooks include complete hyperparameter configurations and can be executed on Kaggle's free tier.

---

## Future Development

### CHAYA-v2 (Planned)

The next iteration will address current architectural limitations:

**Proposed Improvements:**
- Scale to 140-180M parameters for improved capacity
- Grouped Query Attention (GQA) for efficient KV caching
- Extended context window (1024 tokens)
- RoPE base frequency adjustment for better extrapolation
- QK normalization for training stability

**Training Strategy:**
- Multi-stage curriculum: foundation (3.6B tokens) → instruction tuning
- Expanded dataset including RedPajama samples
- Label smoothing and improved regularization

### CHAYA-v3 (Research Direction)

Exploration of advanced techniques:
- Differential attention mechanisms
- Mixture of Experts (MoE) for sparse activation
- Multi-Head Latent Attention (MLA) for memory efficiency


---

## License

This project is released under the MIT License. See LICENSE file for details.

---

## Acknowledgments

**Datasets:**
- WikiText-103: [Merity et al., 2016](https://arxiv.org/abs/1609.07843)
- SWIFT-700 components: OpenWebText, TinyStories, FineWeb-Edu, RedPajama (ArXiv), Dolly-15k, SQuAD

**Architectural Inspiration:**
- RoPE: [Su et al., 2021](https://arxiv.org/abs/2104.09864)
- RMSNorm: [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)
- SwiGLU: [Shazeer, 2020](https://arxiv.org/abs/2002.05202)

**Infrastructure:**
- Training: Kaggle (2× NVIDIA T4 GPUs)
- Tokenization: HuggingFace Transformers library

---

## Contact

For questions, issues, or collaboration inquiries, please open an issue on the GitHub repository.

**Project Page:** https://github.com/Pranoy71/Chaya-v1
**Dataset:** https://github.com/Pranoy71/Swift-700

---

**Note:** This is a research and educational project. The model exhibits typical limitations of small-scale language models including factual inaccuracies and inconsistent reasoning. It is not recommended for production use without extensive evaluation and safety testing.
