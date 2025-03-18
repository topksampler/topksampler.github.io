import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import '../styles/JourneySection.css';

interface JourneySectionProps {
  onContentView: (category?: string) => void;
}

interface JourneyStep {
  id: string;
  title: string;
  content: string;
  ascii: string;
  code?: string;
}

const steps: JourneyStep[] = [
  {
    id: 'concepts',
    title: 'model.train()',
    content: 'Dive deep into neural architectures. From backprop to attention mechanisms, we\'ll optimize your learning rate.',
    ascii: `
    ┌───────────┐
    │ℒ = ∑ℒᵢ/n │
    │[▓▓▓▓░░░░]│
    │epoch: 42 │
    └───────────┘
    `,
    code: 'accuracy = model.fit(X_train, y_train, epochs=42)'
  },
  {
    id: 'tutorials',
    title: 'model.deploy()',
    content: 'Transform theory into production-ready code. Each tutorial is version controlled and tested against reality.',
    ascii: `
    ┌──────────┐
    │git push  │
    │├─MLOps──┤│
    │└─CI/CD──┘│
    │[✓]ready  │
    └──────────┘
    `,
    code: 'docker run -d caffeine-brain:latest'
  },
  {
    id: 'projects',
    title: 'github.commit()',
    content: 'Open source projects where algorithms meet real problems. PRs welcome, bugs expected, learning guaranteed.',
    ascii: `
    ┌─────────┐
    │ ⎇ main  │
    │ └→feat  │
    │   └→fix │
    └─────────┘
    `,
    code: 'git checkout -b feature/neural-magic'
  },
  {
    id: 'thoughts',
    title: 'brain.think()',
    content: 'Where silicon meets neurons. Exploring the space between mathematical elegance and biological chaos.',
    ascii: `
    ╭─────────╮
    │δ(∂L/∂w) │
    │ ⟨ϕ|ψ⟩   │
    │ℝⁿ → ℝᵐ  │
    ╰─────────╯
    `,
    code: 'consciousness = undefined'
  }
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.3
    }
  }
};

const stepVariants = {
  hidden: { 
    opacity: 0, 
    y: 50,
    filter: 'blur(10px)'
  },
  visible: { 
    opacity: 1, 
    y: 0,
    filter: 'blur(0px)',
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 12
    }
  }
};

const codeVariants = {
  hidden: { opacity: 0, x: -20 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: {
      type: "spring",
      stiffness: 100
    }
  }
};

const titleVariants = {
  hidden: { opacity: 0, y: -30 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 15
    }
  }
};

const decorationVariants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { 
    opacity: 1, 
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 10
    }
  }
};

const JourneySection: React.FC<JourneySectionProps> = ({ onContentView }) => {
  const sectionRef = useRef<HTMLElement>(null);

  const handleStepClick = (stepId: string) => {
    onContentView(stepId);
  };

  return (
    <section id="journey" ref={sectionRef} className="journey-section">
      {/* Background elements */}
      <div className="journey-background">
        <div className="grid-overlay"></div>
        <div className="noise-overlay"></div>
        <div className="vignette-overlay"></div>
      </div>
      
      <motion.div 
        className="journey-decoration top-left"
        variants={decorationVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        ╔═══
      </motion.div>
      <motion.div 
        className="journey-decoration top-right"
        variants={decorationVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        ═══╗
      </motion.div>
      
      <motion.div 
        className="journey-intro"
        variants={titleVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <h2 className="journey-title">brain.map<span className="function-parens">()</span></h2>
        <div className="subtitle-container">
          <p className="journey-subtitle">traverse<span className="function-parens">(</span>neural_pathways<span className="function-parens">)</span> <span className="arrow">{'=>'}</span> knowledge</p>
        </div>
        <div className="ascii-progress-container">
          <div className="ascii-progress">
            <span className="progress-label">loading neural pathways:</span>
            <span className="progress-wrapper">[<span className="progress-bar">■■■■■■■■■■</span>]</span>
            <span className="progress-percent">100%</span>
          </div>
        </div>
      </motion.div>
      
      <motion.div 
        className="journey-steps"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
      >
        {steps.map((step, index) => (
          <motion.div
            key={step.id}
            className="journey-step"
            variants={stepVariants}
            onClick={() => handleStepClick(step.id)}
            whileHover={{ 
              y: -5,
              transition: { type: "spring", stiffness: 300 }
            }}
          >
            <div className="step-index">{`0${index + 1}`}</div>
            <div className="step-content">
              <h3 className="step-title">{step.title}</h3>
              <p className="step-description">{step.content}</p>
              {step.code && (
                <motion.div 
                  className="step-code"
                  variants={codeVariants}
                >
                  <div className="code-header">
                    <span className="code-dot"></span>
                    <span className="code-dot"></span>
                    <span className="code-dot"></span>
                    <span className="code-filename">execute.sh</span>
                  </div>
                  <code>{step.code}</code>
                </motion.div>
              )}
              <div className="step-hover-hint">Click to explore</div>
            </div>
            <motion.pre 
              className="step-ascii"
              whileHover={{ 
                scale: 1.05,
                color: "var(--color-accent)",
                transition: { type: "spring", stiffness: 300 }
              }}
            >
              {step.ascii}
            </motion.pre>
            <div className="step-glow"></div>
          </motion.div>
        ))}
      </motion.div>

      <motion.div 
        className="journey-cta"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ delay: 0.8, duration: 0.6 }}
      >
        <motion.button 
          onClick={() => onContentView()}
          whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
          whileTap={{ scale: 0.95 }}
        >
          <span className="button-text">explore.articles()</span>
          <span className="button-glow"></span>
        </motion.button>
        <div className="cta-decoration">
          <div className="cta-line"></div>
          <div className="cta-dots">
            <span className="cta-dot"></span>
            <span className="cta-dot"></span>
            <span className="cta-dot"></span>
          </div>
        </div>
      </motion.div>
      
      <motion.div 
        className="journey-decoration bottom-left"
        variants={decorationVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        ╚═══
      </motion.div>
      <motion.div 
        className="journey-decoration bottom-right"
        variants={decorationVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        ═══╝
      </motion.div>
    </section>
  );
};

export default JourneySection; 