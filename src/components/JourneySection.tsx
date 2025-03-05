import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import '../styles/JourneySection.css';

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

interface JourneySectionProps {
  onContentView: (category?: string) => void;
}

const JourneySection: React.FC<JourneySectionProps> = ({ onContentView }) => {
  const sectionRef = useRef<HTMLElement>(null);
  const [isNavigationEnabled, setIsNavigationEnabled] = useState(true);

  const handleStepClick = (stepId: string) => {
    if (isNavigationEnabled) {
      onContentView(stepId);
    }
  };

  return (
    <section id="journey" ref={sectionRef} className="journey-section">
      <motion.div 
        className="journey-intro"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="journey-title">brain.map()</h2>
        <p className="journey-subtitle">traverse(neural_pathways) ={'>'} knowledge</p>
        <div className="navigation-toggle">
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={isNavigationEnabled}
              onChange={(e) => setIsNavigationEnabled(e.target.checked)}
            />
            <span className="toggle-slider"></span>
          </label>
          <span className="toggle-label">
            navigation.{isNavigationEnabled ? 'enabled()' : 'disabled()'}
          </span>
        </div>
      </motion.div>
      
      <motion.div 
        className="journey-steps"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        {steps.map((step, index) => (
          <motion.div
            key={step.id}
            className={`journey-step ${isNavigationEnabled ? 'clickable' : ''}`}
            variants={stepVariants}
            onClick={() => handleStepClick(step.id)}
          >
            <div className="step-content">
              <h3 className="step-title">{step.title}</h3>
              <p className="step-description">{step.content}</p>
              {step.code && (
                <motion.div 
                  className="step-code"
                  variants={codeVariants}
                >
                  <code>{step.code}</code>
                </motion.div>
              )}
            </div>
            <motion.pre 
              className="step-ascii"
              whileHover={{ 
                scale: 1.05,
                transition: { type: "spring", stiffness: 300 }
              }}
            >
              {step.ascii}
            </motion.pre>
          </motion.div>
        ))}
      </motion.div>

      <motion.div 
        className="journey-cta"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ delay: 0.8 }}
      >
        <button onClick={() => onContentView()}>
          explore.articles()
        </button>
      </motion.div>
    </section>
  );
};

export default JourneySection; 