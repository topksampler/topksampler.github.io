---
id: mlp
title: "mlp.build()"
category: "concepts"
date: "2024-03-21"
description: "Building neural networks from scratch with PyTorch"
readingTime: "10 min"
ascii: |
  ┌─[o]─[o]─┐
  │   [o]   │
  │ [o]─[o] │
  └─────────┘
---

# Multi-Layer Perceptrons

Let's build a neural network from scratch and understand every component, from forward propagation to backpropagation.

## Architecture

The basic building block of neural networks is the perceptron. When we stack multiple layers of perceptrons, we get an MLP:

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.layer1(x))
        return self.layer2(x)
```

## Training Loop

Here's how we train our MLP:

```python
def train(model, dataloader, criterion, optimizer):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

## Activation Functions

The choice of activation function is crucial. ReLU is popular, but there are others:

```python
# ReLU
def relu(x):
    return max(0, x)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh
def tanh(x):
    return np.tanh(x)
``` 