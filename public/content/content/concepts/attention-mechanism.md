---
id: attention
title: attention.explain()
category: concepts
description: Deep dive into transformer attention mechanisms. From scaled dot-product to multi-head attention.
date: 2024-03-15
readingTime: 15 min
ascii: |
  ┌─Q──K──V─┐
  │ ▲  ▲  ▲ │
  │ └┼──┼─┘ │
  │  └──┘   │
  └─────────┘
---

# Intro

Attention mechanisms have revolutionized how neural networks process sequential data. In this deep dive, we'll explore the mathematics and intuition behind attention, from its basic principles to advanced implementations.

## The Intuition

At its core, attention is about learning which parts of the input are most relevant for each part of the output. Just as humans focus on specific words when reading a sentence, attention mechanisms allow neural networks to 'focus' on relevant parts of the input sequence.

## Scaled Dot-Product Attention

The fundamental building block of modern attention mechanisms is scaled dot-product attention. It computes the compatibility between queries and keys, then uses these scores to create a weighted sum of values.

```python
def scaled_dot_product_attention(Q, K, V):
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # Scale the scores
    d_k = torch.tensor(K.size(-1))
    scores = scores / torch.sqrt(d_k)
    # Apply softmax to get probabilities
    attention = torch.softmax(scores, dim=-1)
    # Compute weighted sum of values
    return torch.matmul(attention, V)
```

## Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces. Each head can learn to focus on different aspects of the input, such as syntax, semantics, or positional relationships.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
```

# Conclusion

Understanding attention mechanisms is crucial for working with modern neural architectures. They've become the foundation for state-of-the-art models in NLP, computer vision, and beyond. 