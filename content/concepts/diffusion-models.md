---
id: diffusion
title: "diffusion.explain()"
category: "concepts"
date: "2024-03-20"
description: "Understanding the mathematics and intuition behind diffusion models"
readingTime: "12 min"
ascii: |
  ┌─NOISE───┐
  │ ▒▓░▒▓░▒ │
  │ ░▒▓░▒▓░ │
  │ ▓░▒▓░▒▓ │
  └─────────┘
---

# Understanding Diffusion Models

Diffusion models have revolutionized generative AI, enabling the creation of highly realistic images, audio, and more. Let's dive deep into how they work.

## The Intuition

At their core, diffusion models work by gradually adding noise to data and then learning to reverse this process. It's like slowly turning an image into static, then learning to recover the original image from the noise.

## The Mathematics

The diffusion process can be described by a forward process q(x_t|x_{t-1}) and a reverse process p(x_{t-1}|x_t). The forward process gradually adds Gaussian noise, while the reverse process learns to denoise.

```python
def diffusion_forward(x_0, t):
    """Forward diffusion process"""
    noise = torch.randn_like(x_0)
    alpha_t = get_alpha(t)
    return torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
```

## Training Process

The model is trained to predict the noise added at each step, allowing it to learn the reverse process:

```python
def train_step(model, x_0):
    t = torch.randint(0, T, (x_0.shape[0],))
    noise = torch.randn_like(x_0)
    noisy_x = add_noise(x_0, noise, t)
    predicted_noise = model(noisy_x, t)
    loss = F.mse_loss(predicted_noise, noise)
    return loss
```

## Sampling

During inference, we start with pure noise and gradually denoise to generate samples:

```python
def sample(model, shape):
    x_t = torch.randn(shape)
    for t in reversed(range(T)):
        x_t = denoise_step(model, x_t, t)
    return x_t
``` 