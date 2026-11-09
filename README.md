# Transformer from Scratch

This project contains a PyTorch implementation of a Transformer model built from the ground up, following the original encoder-decoder architecture introduced in the paper "Attention Is All You Need".

The implementation is contained in `Transformer_from_scratch.py` and includes:

- Multi-head self-attention
- Positional encoding
- Encoder and decoder stacks
- Residual connections and layer normalization
- Feed-forward network blocks
- Source and target masking for training/inference

## What this script does

The script defines a complete `Transformer` model class that can be instantiated and used for sequence-to-sequence tasks such as:

- machine translation
- text generation
- sequence classification pipelines
- custom NLP experiments

At a high level, the model:

1. embeds source tokens and adds positional information
2. processes them through an encoder stack
3. builds masked decoder inputs for target sequences
4. attends to both decoder inputs and encoder outputs
5. produces logits over the target vocabulary

## Main components

The script is organized into the following core parts:

- `SelfAttention`: computes scaled dot-product attention across multiple heads
- `TransformerBlock`: combines attention + feed-forward network + residuals
- `Encoder`: converts input tokens into contextualized representations
- `DecoderBlock`: applies masked self-attention and cross-attention
- `Decoder`: maps token embeddings to target vocabulary predictions
- `Transformer`: wraps the full encoder-decoder model and creates masks

## Requirements

Install the dependencies before running the script:

```bash
pip install torch
```

## Quick start

```python
import torch
from Transformer_from_scratch import Transformer

src_vocab_size = 5000
trg_vocab_size = 5000
src_pad_idx = 0
trg_pad_idx = 0

model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    embed_size=256,
    num_layers=6,
    heads=8,
    forward_expansion=4,
    dropout=0.1,
    device="cpu",
    max_length=100
)

src = torch.randint(1, src_vocab_size, (32, 20))
trg = torch.randint(1, trg_vocab_size, (32, 18))

output = model(src, trg)
print(output.shape)  # Expected: (batch_size, target_length, trg_vocab_size)
```

## Notes

- This file is a model definition and learning resource rather than a full training pipeline.
- For real tasks, you would normally add:
  - data preprocessing/tokenization
  - a training loop
  - optimizer and loss function
  - evaluation metrics
  - batching and padding logic
- The mask logic prevents the decoder from seeing future tokens during training and ignores padding positions.

## Customization ideas

You can tune the model by changing these parameters:

- `embed_size`
- `num_layers`
- `heads`
- `forward_expansion`
- `dropout`
- `max_length`

These values control the model capacity and training behavior for different dataset sizes and tasks.

## License

This project does not include a license file. If you plan to reuse or distribute it, add an appropriate license that matches your project requirements.
