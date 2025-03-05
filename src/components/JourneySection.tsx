import { useEffect, useRef } from 'react';
import '../styles/JourneySection.css';

interface JourneyStep {
  id: string;
  title: string;
  content: string;
  ascii: string;
}

const steps: JourneyStep[] = [
  {
    id: 'future',
    title: 'The Future is Now',
    content: 'AI isn\'t just coming - it\'s here. Understanding AI isn\'t optional anymore; it\'s becoming as fundamental as digital literacy was in the past decades.',
    ascii: `
    ╭───────────╮
    │  20XX     │
    │    ▲      │
    │ YOU ARE   │
    │   HERE    │
    ╰───────────╯
    `
  },
  {
    id: 'empowerment',
    title: 'Personal Empowerment',
    content: 'Knowledge of AI empowers you to make informed decisions, understand its capabilities and limitations, and harness its potential for your own growth.',
    ascii: `
      ╱|\\
    ╱__|_\\
    │ AI │
    │YOU │
    ‾‾‾‾‾
    `
  },
  {
    id: 'creativity',
    title: 'Augmented Creativity',
    content: 'AI isn\'t here to replace human creativity - it\'s here to augment it. Learn how to dance with algorithms and create something truly unique.',
    ascii: `
    ╭─╮ ╭─╮
    │H│~│A│
    │U│~│I│
    ╰─╯ ╰─╯
    SYNERGY
    `
  },
  {
    id: 'responsibility',
    title: 'Ethical Responsibility',
    content: 'As AI becomes more prevalent, understanding its ethical implications becomes crucial. Be part of the conversation that shapes its future.',
    ascii: `
    ╔════════╗
    ║ ETHICS ║
    ║  ┌─┐   ║
    ║  │A│   ║
    ║  │I│   ║
    ║  └─┘   ║
    ╚════════╝
    `
  }
];

const JourneySection = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const stepsRef = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.2 }
    );

    stepsRef.current.forEach((step) => {
      if (step) observer.observe(step);
    });

    return () => observer.disconnect();
  }, []);

  const setStepRef = (el: HTMLDivElement | null, index: number) => {
    stepsRef.current[index] = el;
  };

  return (
    <section id="journey" ref={sectionRef} className="journey-section">
      <div className="journey-intro">
        <h2 className="journey-title">Why AI Matters</h2>
        <p className="journey-subtitle">Understanding the path forward in an AI-driven world</p>
      </div>
      
      <div className="journey-steps">
        {steps.map((step, index) => (
          <div
            key={step.id}
            ref={(el) => setStepRef(el, index)}
            className="journey-step"
          >
            <div className="step-content">
              <h3 className="step-title">{step.title}</h3>
              <p className="step-description">{step.content}</p>
            </div>
            <pre className="step-ascii">{step.ascii}</pre>
          </div>
        ))}
      </div>

      <div className="journey-cta">
        <button onClick={() => document.getElementById('blog')?.scrollIntoView({ behavior: 'smooth' })}>
          explore.articles()
        </button>
      </div>
    </section>
  );
};

export default JourneySection; 