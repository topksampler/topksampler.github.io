import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import { watchContent, ContentData } from '../utils/contentLoader';
import '../styles/ContentBrowser.css';

interface ContentBrowserProps {
  initialCategory: string | null;
  articleId?: string | null;
  onBack: () => void;
}

interface Section {
  title: string;
  content: string;
  code?: string;
}

interface ContentNode {
  id: string;
  title: string;
  category: 'concepts' | 'tutorials' | 'projects' | 'thoughts';
  description: string;
  date: string;
  readingTime: string;
  ascii: string;
  content: {
    intro: string;
    sections: {
      title: string;
      content: string;
      code?: string;
    }[];
    conclusion?: string;
  };
}

const sampleContent: ContentNode[] = [
  // Concepts
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
          content: "At its core, attention is about learning which parts of the input are most relevant for each part of the output. Just as humans focus on specific words when reading a sentence, attention mechanisms allow neural networks to 'focus' on relevant parts of the input sequence.",
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
        },
        {
          title: "3. Multi-Head Attention",
          content: "Multi-head attention allows the model to jointly attend to information from different representation subspaces. Each head can learn to focus on different aspects of the input, such as syntax, semantics, or positional relationships.",
          code: `class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)`
        }
      ],
      conclusion: "Understanding attention mechanisms is crucial for working with modern neural architectures. They've become the foundation for state-of-the-art models in NLP, computer vision, and beyond."
    }
  },
  {
    id: 'backprop',
    title: 'gradient.flow()',
    category: 'concepts',
    description: 'Understanding backpropagation through computational graphs.',
    date: '2024-03-10',
    readingTime: '12 min',
    ascii: `
    ┌─∂L/∂w───┐
    │ ↙ ↓ ↘   │
    │∂x ∂y ∂z │
    └─────────┘`,
    content: {
      intro: "Backpropagation is the cornerstone of deep learning, enabling neural networks to learn from their mistakes. Let's break down this elegant algorithm and understand how gradients flow through computational graphs.",
      sections: [
        {
          title: "1. The Chain Rule",
          content: "At its heart, backpropagation is an application of the chain rule from calculus. We'll see how this fundamental principle allows us to compute gradients efficiently through nested functions.",
          code: `def backward_pass(computational_graph):
    # Initialize gradients
    gradients = {}
    for node in reversed(computational_graph):
        # Apply chain rule
        grad = node.output_grad * node.local_grad
        gradients[node] = grad
    return gradients`
        },
        {
          title: "2. Computational Graphs",
          content: "Neural networks can be represented as directed acyclic graphs (DAGs). Each node represents an operation, and edges represent the flow of data and gradients.",
          code: `class Node:
    def __init__(self):
        self.inputs = []
        self.output = None
        self.gradients = {}
        
    def forward(self):
        # Compute output
        pass
        
    def backward(self, gradient):
        # Distribute gradient to inputs
        pass`
        },
        {
          title: "3. Automatic Differentiation",
          content: "Modern deep learning frameworks implement reverse-mode automatic differentiation, making it possible to compute gradients automatically for arbitrary computational graphs.",
          code: `import torch

x = torch.tensor([1.0], requires_grad=True)
y = x * 2
z = y ** 2
z.backward()
print(x.grad)  # dz/dx = 4.0`
        }
      ],
      conclusion: "Understanding backpropagation deeply is crucial for developing intuition about neural network training and debugging optimization issues."
    }
  },
  // Tutorials
  {
    id: 'llm-deployment',
    title: 'llm.deploy()',
    category: 'tutorials',
    description: 'Production deployment of LLMs using Docker and FastAPI.',
    date: '2024-03-08',
    readingTime: '20 min',
    ascii: `
    ┌─Docker──┐
    │[LLM]    │
    │├─API──┤ │
    └────────┘`,
    content: {
      intro: "Deploying Large Language Models (LLMs) in production requires careful consideration of performance, scalability, and resource management. This guide walks through creating a production-ready LLM API service.",
      sections: [
        {
          title: "1. Environment Setup",
          content: "First, we'll set up our development environment with the necessary dependencies. We'll use Python 3.9+ for compatibility with modern ML libraries.",
          code: `# requirements.txt
fastapi==0.68.0
uvicorn==0.15.0
transformers==4.30.2
torch==2.0.1
python-dotenv==1.0.0`
        },
        {
          title: "2. FastAPI Implementation",
          content: "We'll create a FastAPI application that serves our LLM. The API will include endpoints for text generation, embeddings, and model information.",
          code: `from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="LLM API")

class LLMService:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "model_name",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("model_name")

@app.post("/generate")
async def generate_text(prompt: str):
    try:
        tokens = llm.tokenizer(prompt, return_tensors="pt")
        output = llm.model.generate(**tokens, max_length=100)
        return {"text": llm.tokenizer.decode(output[0])}
    except Exception as e:
        raise HTTPException(status_code=500, str(e))`
        },
        {
          title: "3. Docker Configuration",
          content: "Containerization ensures consistent deployment across different environments. Our Docker setup includes CUDA support for GPU acceleration.",
          code: `# Dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]`
        }
      ],
      conclusion: "With this setup, you have a production-ready LLM service that can scale horizontally and handle high-throughput requests efficiently."
    }
  },
  {
    id: 'data-pipeline',
    title: 'data.transform()',
    category: 'tutorials',
    description: 'Building robust ML data pipelines with Apache Beam.',
    date: '2024-03-05',
    readingTime: '18 min',
    ascii: `
    ┌→ ⚡ →┐
    │DATA  │
    │FLOW  │
    └──────┘`,
    content: {
      intro: "Learn how to build scalable and maintainable data pipelines for machine learning using Apache Beam. We'll cover best practices for data preprocessing, transformation, and validation.",
      sections: [
        {
          title: "1. Apache Beam Basics",
          content: "Apache Beam provides a unified programming model for batch and streaming data processing. We'll explore its core concepts and how to implement them.",
          code: `import apache_beam as beam

def preprocess_data(element):
    # Transform data
    return transformed_element

with beam.Pipeline() as pipeline:
    data = (pipeline 
            | 'Read' >> beam.io.ReadFromText('input.txt')
            | 'Transform' >> beam.Map(preprocess_data)
            | 'Write' >> beam.io.WriteToText('output.txt'))`
        },
        {
          title: "2. Data Validation",
          content: "Implement robust data validation to ensure data quality and catch issues early in the pipeline.",
          code: `from tensorflow_data_validation as tfdv

stats = tfdv.generate_statistics_from_dataframe(df)
schema = tfdv.infer_schema(stats)
anomalies = tfdv.validate_statistics(stats, schema)

if anomalies.anomaly_info:
    raise ValueError(f"Data anomalies found: {anomalies}")`
        }
      ],
      conclusion: "With these patterns in place, you'll have a robust foundation for building production-ready ML data pipelines that can scale with your needs."
    }
  },
  // Projects
  {
    id: 'neural-search',
    title: 'vector.search()',
    category: 'projects',
    description: 'Open-source neural search engine built with FAISS.',
    date: '2024-03-01',
    readingTime: '10 min',
    ascii: `
    ┌─[·]────┐
    │↗ ↑ ↖   │
    │← · →   │
    │↙ ↓ ↘   │
    └────────┘`,
    content: {
      intro: "Building a scalable neural search engine using FAISS for efficient similarity search in high-dimensional spaces. This project demonstrates how to create a production-ready vector search system.",
      sections: [
        {
          title: "1. Vector Embeddings",
          content: "We'll use transformer models to convert text into high-dimensional vectors that capture semantic meaning. These embeddings form the foundation of our search system.",
          code: `from transformers import AutoModel, AutoTokenizer

def get_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    tokens = tokenizer(texts, padding=True, truncation=True,
                      return_tensors="pt")
    outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1)`
        },
        {
          title: "2. FAISS Index",
          content: "FAISS provides efficient similarity search and clustering of dense vectors. We'll create an index optimized for our embedding dimension and dataset size.",
          code: `import faiss

def create_index(vectors, dimension=384):
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

def search(index, query_vector, k=5):
    distances, indices = index.search(query_vector, k)
    return distances, indices`
        }
      ],
      conclusion: "This neural search engine provides fast and accurate semantic search capabilities, scaling to millions of documents while maintaining sub-second query times."
    }
  },
  {
    id: 'edge-ml',
    title: 'edge.optimize()',
    category: 'projects',
    description: 'TinyML: Running neural networks on microcontrollers.',
    date: '2024-02-28',
    readingTime: '14 min',
    ascii: `
    ┌─μCTRL──┐
    │NN-{'>'}BIN │
    │[32KB]  │
    └────────┘`,
    content: {
      intro: "Deploying neural networks on microcontrollers requires careful optimization and quantization. Learn how to compress models to run on devices with limited memory and processing power.",
      sections: [
        {
          title: "1. Model Quantization",
          content: "Convert floating-point weights to 8-bit integers while preserving model accuracy. This reduces model size and improves inference speed on embedded devices.",
          code: `import tensorflow as tf

def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    tflite_model = converter.convert()
    return tflite_model`
        },
        {
          title: "2. Memory Optimization",
          content: "Optimize memory usage by implementing efficient buffer management and reducing activation memory requirements.",
          code: `// Arduino implementation
#define TENSOR_ARENA_SIZE 32*1024
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

void setup() {
  tflite::MicroErrorReporter error_reporter;
  tflite::AllOpsResolver resolver;
  
  const tflite::Model* model = 
    tflite::GetModel(g_model);
  tflite::MicroInterpreter interpreter(
    model, resolver, tensor_arena,
    TENSOR_ARENA_SIZE, &error_reporter);
}`
        }
      ],
      conclusion: "With these optimizations, we can run sophisticated neural networks on devices with as little as 32KB of memory, enabling AI at the edge."
    }
  },
  // Thoughts
  {
    id: 'ai-creativity',
    title: 'mind.ponder()',
    category: 'thoughts',
    description: 'Exploring the nature of artificial creativity.',
    date: '2024-02-25',
    readingTime: '8 min',
    ascii: `
    ╭─⚡─☆─⚡─╮
    │HUMAN AI│
    │SPARK++│
    ╰───────╯`,
    content: {
      intro: "What does it mean for an AI to be creative? This exploration delves into the intersection of human creativity and machine learning, questioning our assumptions about artificial creativity.",
      sections: [
        {
          title: "1. The Creative Process",
          content: "Creativity isn't just about generating novel outputs—it's about understanding context, making unexpected connections, and producing meaningful innovations. How do neural networks mirror or differ from human creative processes?",
        },
        {
          title: "2. Emergent Behaviors",
          content: "Large language models and generative AI systems often exhibit unexpected behaviors that appear creative. Are these genuine innovations or sophisticated pattern matching?",
          code: `# Example of emergent behavior
response = llm.generate(
  "Write a haiku about neural networks
   in the style of a quantum physicist")`
        }
      ],
      conclusion: "The boundary between human and artificial creativity continues to blur, challenging our understanding of both intelligence and creativity itself."
    }
  },
  {
    id: 'future-ml',
    title: 'future.predict()',
    category: 'thoughts',
    description: 'Speculative directions in machine learning.',
    date: '2024-02-20',
    readingTime: '9 min',
    ascii: `
    ┌─2025───┐
    │↗↑↗↑↗↑↗│
    │ML→∞   │
    └────────┘`,
    content: {
      intro: "As machine learning evolves at an unprecedented pace, what paradigms will emerge in the coming years? Let's explore potential futures of AI and their implications.",
      sections: [
        {
          title: "1. Neuromorphic Computing",
          content: "Future AI systems might more closely mimic biological neural networks, with analog computing elements and spike-based processing. This could lead to more efficient and adaptable systems.",
        },
        {
          title: "2. Quantum Machine Learning",
          content: "The intersection of quantum computing and machine learning promises new approaches to optimization and pattern recognition.",
          code: `# Speculative quantum neural network
class QuantumNeuralNetwork:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.quantum_state = initialize_superposition()
    
    def quantum_layer(self, params):
        # Apply quantum gates
        return apply_unitary(self.quantum_state, params)`
        }
      ],
      conclusion: "The future of machine learning will likely transcend current paradigms, incorporating insights from neuroscience, physics, and emerging computing architectures."
    }
  }
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { 
    opacity: 1,
    transition: { 
      staggerChildren: 0.05,
      delayChildren: 0.1
    }
  },
  exit: { opacity: 0 }
};

const cardVariants = {
  hidden: { 
    opacity: 0,
    y: 20
  },
  visible: { 
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 15
    }
  },
  exit: { 
    opacity: 0,
    y: -20,
    transition: {
      duration: 0.2
    }
  },
  hover: {
    y: -4,
    transition: {
      type: "spring",
      stiffness: 200,
      damping: 20
    }
  }
};

const overlayVariants = {
  hidden: { opacity: 0 },
  visible: { 
    opacity: 1,
    transition: {
      duration: 0.2
    }
  }
};

const modalVariants = {
  hidden: {
    opacity: 0,
    scale: 0.8,
    y: 20
  },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 25
    }
  },
  exit: {
    opacity: 0,
    scale: 0.8,
    y: 20,
    transition: {
      duration: 0.2
    }
  }
};

const conceptsAscii = `
┌──────────────┐
│ ▄▄▄▄▄▄▄▄▄▄▄ │
│ █ AI/ML    █ │
│ █ THEORY   █ │
│ █▀▀▀▀▀▀▀▀▀█ │
└──────────────┘`;

const tutorialsAscii = `
┌──────────────┐
│ [Docker]     │
│ ├── LLM     │
│ └── API     │
│ System.run() │
└──────────────┘`;

const projectsAscii = `
┌──────────────┐
│ ┌─Projects─┐ │
│ │ ▲  ▲  ▲ │ │
│ │ └──┼──┘ │ │
│ └────┴────┘ │
└──────────────┘`;

const thoughtsAscii = `
┌──────────────┐
│  ⚡️HUMAN⚡️  │
│  ┌──AI──┐   │
│  │SPARK│    │
│  └─────┘++  │
└──────────────┘`;

const categoryDescriptions = {
  concepts: {
    title: 'concepts.explore()',
    description: 'Deep dives into AI/ML theory',
    ascii: conceptsAscii
  },
  tutorials: {
    title: 'tutorials.build()',
    description: 'From theory to production',
    ascii: tutorialsAscii
  },
  projects: {
    title: 'projects.create()',
    description: 'Open source AI adventures',
    ascii: projectsAscii
  },
  thoughts: {
    title: 'thoughts.spark()',
    description: 'Neural musings && human insights',
    ascii: thoughtsAscii
  }
};

const MatrixRain: React.FC = () => {
  const [characters, setCharacters] = useState<Array<{id: number; char: string; x: number; delay: number}>>([]);

  useEffect(() => {
    const matrixChars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンヴ';
    const numCharacters = 50;
    const newCharacters = Array.from({ length: numCharacters }, (_, i) => ({
      id: i,
      char: matrixChars[Math.floor(Math.random() * matrixChars.length)],
      x: Math.random() * 100,
      delay: Math.random() * 2
    }));
    setCharacters(newCharacters);
  }, []);

  return (
    <div className="matrix-rain">
      {characters.map(char => (
        <div
          key={char.id}
          className="matrix-character"
          style={{
            left: `${char.x}%`,
            animationDelay: `${char.delay}s`
          }}
        >
          {char.char}
        </div>
      ))}
    </div>
  );
};

const TerminalHeader: React.FC<{ title: string }> = ({ title }) => (
  <div className="terminal-header">
    <div className="terminal-controls">
      <span className="control close"></span>
      <span className="control minimize"></span>
      <span className="control maximize"></span>
    </div>
    <div className="terminal-title">{title}</div>
    <div className="terminal-cursor">_</div>
  </div>
);

const ContentBrowser: React.FC<ContentBrowserProps> = ({ initialCategory, articleId, onBack }) => {
  const { category } = useParams<{ category: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const [selectedArticle, setSelectedArticle] = useState<ContentData | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(category || initialCategory);
  const [content, setContent] = useState<ContentData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Define the CaffeineBrain ASCII art from the homepage
  const caffeineAsciiArt = `
   ██████╗ █████╗ ███████╗███████╗███████╗██╗███╗   ██╗███████╗
  ██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔════╝
  ██║     ███████║█████╗  █████╗  █████╗  ██║██╔██╗ ██║█████╗  
  ██║     ██╔══██║██╔══╝  ██╔══╝  ██╔══╝  ██║██║╚██╗██║██╔══╝  
  ╚██████╗██║  ██║██║     ██║     ███████╗██║██║ ╚████║███████╗
   ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝
                                                          
  ██████╗ ██████╗  █████╗ ██╗███╗   ██╗                        
  ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                        
  ██████╔╝██████╔╝███████║██║██╔██╗ ██║                        
  ██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║                        
  ██████╔╝██║  ██║██║  ██║██║██║ ╚████║                        
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝                        
`;

  // Define the CB ASCII art logo
  const cbAsciiArt = `
   ██████╗██████╗ 
  ██╔════╝██╔══██╗
  ██║     ██████╔╝
  ██║     ██╔══██╗
  ╚██████╗██████╔╝
   ╚═════╝╚═════╝ 
`;

  useEffect(() => {
    const loadContent = async () => {
      setIsLoading(true);
      setError(null);
      try {
        await watchContent((newContent) => {
          setContent(newContent);
          setIsLoading(false);
        });
      } catch (err) {
        setError('Failed to load content. Please try again later.');
        setIsLoading(false);
      }
    };

    loadContent();
  }, []);

  useEffect(() => {
    setSelectedCategory(category || initialCategory);
  }, [category, initialCategory]);

  useEffect(() => {
    if (articleId && content.length > 0) {
      const article = content.find(item => item.id === articleId);
      if (article) {
        setSelectedArticle(article);
        if (article.category !== selectedCategory) {
          setSelectedCategory(article.category);
        }
      } else {
        setError(`Article "${articleId}" not found`);
        navigate('/content');
      }
    } else if (!articleId) {
      setSelectedArticle(null);
    }
  }, [articleId, content, navigate, selectedCategory]);

  useEffect(() => {
    // Default to 'concepts' when no category is selected
    if (!selectedCategory && !articleId) {
      navigate('/content/concepts');
      setSelectedCategory('concepts');
    }
  }, [selectedCategory, articleId, navigate]);

  // Remove the mouse tracking effect
  useEffect(() => {
    const cards = document.querySelectorAll('.content-card');
    const grid = document.querySelector('.content-grid');

    if (grid) {
      cards.forEach((card) => {
        (card as HTMLElement).style.transform = 'none';
      });
    }
  }, [selectedCategory]);

  const handleCardClick = (article: ContentData) => {
    setSelectedArticle(article);
    navigate(`/content/${article.category}/${article.id}`);
  };

  const handleBack = () => {
    if (location.pathname.includes('/content/')) {
      const pathParts = location.pathname.split('/');
      if (pathParts.length > 3) {
        // If we're in an article, go back to the category
        navigate(`/content/${pathParts[2]}`);
        setSelectedArticle(null);
      } else {
        // If we're in a category, go back to home
        navigate('/');
      }
    } else {
      navigate('/');
    }
  };

  const handleCategoryClick = (newCategory: string | null) => {
    if (newCategory) {
      navigate(`/content/${newCategory}`);
    } else {
      navigate('/content');
    }
    setSelectedCategory(newCategory);
    setSelectedArticle(null);
  };

  const filteredContent = selectedCategory
    ? content.filter(item => item.category === selectedCategory)
    : content;

  const renderArticle = (article: ContentData) => {
    // Convert article content to markdown string
    const markdownContent = `
${article.content.intro}

${article.content.sections.map(section => `
## ${section.title}

${section.content}

${section.code ? `\`\`\`typescript
${section.code}
\`\`\`` : ''}
`).join('\n')}

${article.content.conclusion ? `
## Conclusion

${article.content.conclusion}` : ''}
`;

    return (
      <motion.div
        key="article-view"
        className="article-view"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="article-container">
          {/* <div className="logo-container" style={{ textAlign: 'center', marginBottom: '15px' }}>
            <img src="/brain.svg" alt="Caffeine Brain Logo" style={{ width: '60px', height: 'auto' }} />
          </div> */}
          
          <motion.pre 
            className="article-ascii"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {article.ascii}
          </motion.pre>
          
          <motion.h1 
            className="article-title"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            {article.title}
          </motion.h1>
          
          <motion.div 
            className="article-meta"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <span>{new Date(article.date).toLocaleDateString('en-US', {
              year: 'numeric',
              month: 'long',
              day: 'numeric'
            })}</span>
            <span>{article.readingTime}</span>
          </motion.div>

          <motion.div
            className="markdown-content"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeRaw]}
              components={{
                code({node, inline, className, children, ...props}) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <div style={{ position: 'relative' }}>
                      <SyntaxHighlighter
                        style={atomDark}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                      <button
                        className="copy-button"
                        onClick={async (e) => {
                          const button = e.currentTarget;
                          try {
                            await navigator.clipboard.writeText(String(children));
                            button.innerHTML = `
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M20 6L9 17l-5-5"></path>
                              </svg>
                              <span>Copied!</span>
                            `;
                            setTimeout(() => {
                              button.innerHTML = `
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                </svg>
                                <span>Copy</span>
                              `;
                            }, 2000);
                          } catch (err) {
                            console.error('Failed to copy code:', err);
                            button.innerHTML = `
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="15" y1="9" x2="9" y2="15"></line>
                                <line x1="9" y1="9" x2="15" y2="15"></line>
                              </svg>
                              <span>Error!</span>
                            `;
                          }
                        }}
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <span>Copy</span>
                      </button>
                    </div>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                }
              }}
            >
              {markdownContent}
            </ReactMarkdown>
          </motion.div>
        </div>
      </motion.div>
    );
  };

  if (isLoading) {
    return (
      <div className="content-browser">
        <div className="loading">Loading content...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="content-browser">
        <div className="error">{error}</div>
      </div>
    );
  }

  return (
    <motion.div 
      className="content-browser"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <MatrixRain />
      <motion.button
        className="back-button"
        onClick={handleBack}
        whileHover={{ scale: 1.05, x: -5 }}
        whileTap={{ scale: 0.95 }}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
      >
        cd ..
      </motion.button>

      <AnimatePresence mode="wait">
        {!selectedArticle ? (
          <motion.div
            key="content-grid"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.5 }}
          >
            <motion.div 
              className="browser-header"
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ 
                type: "spring",
                stiffness: 200,
                damping: 20,
                delay: 0.2 
              }}
            >
              <div className="logo-container" style={{ textAlign: 'center', marginBottom: '15px' }}>
                <pre style={{ 
                  fontFamily: 'JetBrains Mono, monospace', 
                  color: 'var(--color-gray-300)',
                  fontSize: '0.7rem',
                  margin: '0 auto',
                  whiteSpace: 'pre',
                  textAlign: 'center'
                }}>
                  {/* {cbAsciiArt} */}
                </pre>
              </div>
              <pre className="ascii-header">
                {caffeineAsciiArt}
              </pre>
            </motion.div>

            <motion.div 
              className="category-filters"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ 
                type: "spring",
                stiffness: 200,
                damping: 20,
                delay: 0.3 
              }}
            >
              {['concepts', 'tutorials', 'projects', 'thoughts'].map((cat, index) => (
                <motion.button
                  key={`category-${cat}`}
                  className={`category-btn ${selectedCategory === cat ? 'active' : ''}`}
                  onClick={() => handleCategoryClick(cat)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ 
                    type: "spring",
                    stiffness: 200,
                    damping: 20,
                    delay: 0.4 + index * 0.1 
                  }}
                >
                  {cat}
                </motion.button>
              ))}
            </motion.div>

            <motion.div 
              className="content-grid"
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
            >
              {filteredContent.map((article, index) => (
                <motion.div
                  key={`${article.id}-card`}
                  className={`content-card ${article.category}`}
                  variants={cardVariants}
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  whileHover="hover"
                  onClick={() => handleCardClick(article)}
                  custom={index}
                  layout
                >
                  <motion.div className="terminal-header">
                    <div className="terminal-controls">
                      <div className="control close"></div>
                      <div className="control minimize"></div>
                      <div className="control maximize"></div>
                    </div>
                    <div className="terminal-title">{article.title}</div>
                  </motion.div>
                  <motion.pre 
                    className="card-ascii"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                  >
                    {article.ascii}
                  </motion.pre>
                  <motion.div 
                    className="card-description"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                  >
                    {article.description}
                  </motion.div>
                  <motion.div 
                    className="card-meta"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                  >
                    <span>
                      {new Date(article.date).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric'
                      })} • {article.readingTime}
                    </span>
                  </motion.div>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>
        ) : (
          renderArticle(selectedArticle)
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ContentBrowser;