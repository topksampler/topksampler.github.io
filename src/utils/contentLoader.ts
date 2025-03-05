import { ContentData } from '../types';

const sampleContent: ContentData[] = [
  {
    id: 'attention-mechanism',
    title: 'attention.explain()',
    category: 'concepts',
    description: 'Deep dive into transformer attention mechanisms. From scaled dot-product to multi-head attention.',
    date: '2024-03-15',
    readingTime: '15 min',
    ascii: `
    ┌─Q──K──V─┐
    │ ▲  ▲  ▲ │
    │ └┼──┼─┘ │
    │  └──┘   │
    └─────────┘`,
    content: {
      intro: "Attention mechanisms have revolutionized how neural networks process sequential data. In this deep dive, we'll explore the mathematics and intuition behind attention, from its basic principles to advanced implementations.",
      sections: [
        {
          title: "1. The Intuition",
          content: "At its core, attention is about learning which parts of the input are most relevant for each part of the output. Just as humans focus on specific words when reading a sentence, attention mechanisms allow neural networks to 'focus' on relevant parts of the input sequence."
        },
        {
          title: "2. Scaled Dot-Product Attention",
          content: "The fundamental building block of modern attention mechanisms is scaled dot-product attention. It computes the compatibility between queries and keys, then uses these scores to create a weighted sum of values.",
          code: `def scaled_dot_product_attention(Q, K, V):
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # Scale the scores
    d_k = torch.tensor(K.size(-1))
    scores = scores / torch.sqrt(d_k)
    # Apply softmax to get probabilities
    attention = torch.softmax(scores, dim=-1)
    # Compute weighted sum of values
    return torch.matmul(attention, V)`
        }
      ],
      conclusion: "Understanding attention mechanisms is crucial for working with modern neural architectures."
    }
  },
  // Add more sample content here as needed
];

export function watchContent(callback: (content: ContentData[]) => void) {
  // Simulate content loading
  setTimeout(() => {
    callback(sampleContent);
  }, 100);
}

export type { ContentData }; 