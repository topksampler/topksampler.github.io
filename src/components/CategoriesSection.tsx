import { useEffect, useRef } from 'react';
import '../styles/CategoriesSection.css';

interface Category {
  title: string;
  description: string;
  ascii: string;
  code: string;
}

const categories: Category[] = [
  {
    title: 'Multi-Layer Perceptrons',
    description: 'The building blocks of neural networks',
    ascii: `
    [Input] → (Hidden) → [Output]
       ↗   ↘   ↗   ↘   ↗   ↘
    [x₁] → (h₁) → [y₁]
       ↘   ↗   ↘   ↗
    [x₂] → (h₂) → [y₂]
    `,
    code: `class MLP:
  def forward(x):
    h = sigmoid(W1 @ x + b1)
    y = softmax(W2 @ h + b2)
    return y`,
  },
  {
    title: 'Transformer Architecture',
    description: 'Attention is all you need',
    ascii: `
    ┌─ Self-Attention ─┐
    │  Q    K    V    │
    │  ↓    ↓    ↓    │
    │ [Scaled Dot-Product]
    │       ↓         │
    └──── Output ─────┘
    `,
    code: `def attention(q, k, v):
  scores = q @ k.T
  weights = softmax(scores)
  return weights @ v`,
  },
  {
    title: 'Neural Networks',
    description: 'Mimicking biological neural systems',
    ascii: `
      Neuron
    ╭─────────╮
 →──┤ Σwixi+b ├──→
    ╰─────────╯
    activation()
    `,
    code: `def neuron(x, w, b):
  z = np.dot(w, x) + b
  return sigmoid(z)`,
  },
  {
    title: 'Deep Learning',
    description: 'Going deeper into neural architectures',
    ascii: `
    Layer 1   Layer 2   Layer 3
      ○         ○         ○
    ○ ○ ○     ○ ○     ○ ○ ○
      ○       ○ ○         ○
    `,
    code: `model = Sequential([
  Dense(128, activation='relu'),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax')
])`,
  },
  {
    title: 'AI Ethics',
    description: 'Responsible AI development',
    ascii: `
    ╔═══ AI Ethics ═══╗
    ║ □ Fairness      ║
    ║ □ Transparency  ║
    ║ □ Privacy      ║
    ║ □ Safety       ║
    ╚═══════════════╝
    `,
    code: `def check_bias(model, data):
  bias_score = fairness_metric(
    model.predict(data),
    data.sensitive_attrs
  )
  return bias_score`,
  },
];

const CategoriesSection = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const cardsRef = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.1 }
    );

    cardsRef.current.forEach((card) => {
      if (card) observer.observe(card);
    });

    return () => observer.disconnect();
  }, []);

  const setCardRef = (el: HTMLDivElement | null, index: number) => {
    cardsRef.current[index] = el;
  };

  return (
    <section ref={sectionRef} className="categories-section">
      {categories.map((category, index) => (
        <div
          key={category.title}
          ref={(el) => setCardRef(el, index)}
          className="category-card"
        >
          <div className="category-content">
            <h2 className="category-title">{category.title}</h2>
            <p className="category-description">{category.description}</p>
            <pre className="category-ascii">{category.ascii}</pre>
            <pre className="category-code">{category.code}</pre>
          </div>
        </div>
      ))}
    </section>
  );
};

export default CategoriesSection; 