import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import '../styles/HeroSection.css';
import SocialLinks from './SocialLinks';

interface HeroSectionProps {
  onContentView: (category?: string) => void;
}

const HeroSection = ({ onContentView }: HeroSectionProps) => {
  const [typedText, setTypedText] = useState('');
  const fullText = '> exploring human x machine creativity';
  const asciiArt = `
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

  const tags = [
    { id: 'concepts', name: 'Concepts', desc: 'Deep Dives into AI/ML Theory' },
    { id: 'tutorials', name: 'Tutorials', desc: 'From Theory to Production' },
    { id: 'projects', name: 'Projects', desc: 'Open Source AI Adventures' },
    { id: 'thoughts', name: 'Thoughts', desc: 'Neural Musings && Human Insights' }
  ];

  useEffect(() => {
    let currentIndex = 0;
    const typingInterval = setInterval(() => {
      if (currentIndex < fullText.length) {
        setTypedText(fullText.substring(0, currentIndex + 1));
        currentIndex++;
      } else {
        clearInterval(typingInterval);
      }
    }, 50);

    return () => clearInterval(typingInterval);
  }, []);

  return (
    <div className="dark-panel">
      <section className="hero-section">
        {/* Background elements */}
        <div className="hero-background">
          <div className="grid-overlay"></div>
          <div className="glitch-effect"></div>
          <div className="vignette-overlay"></div>
        </div>

        <motion.div
          className="hero-content"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
        >
          <motion.pre
            className="ascii-art"
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            {asciiArt}
          </motion.pre>

          <motion.div
            className="terminal"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <div className="terminal-header">
              <div className="terminal-controls">
                <span className="control close"></span>
                <span className="control minimize"></span>
                <span className="control maximize"></span>
              </div>
              <div className="terminal-title">caffeine_brain.sh</div>
            </div>
            <div className="fixed-width-container">
              <div className="terminal-text-container">
                <p className="typed-text">{typedText}</p>
                <span className="cursor" />
              </div>
            </div>
          </motion.div>

          <motion.div
            className="topics-container"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <div className="topics">
              {tags.map((tag, index) => (
                <motion.div
                  key={tag.id}
                  className="topic-wrapper"
                  onClick={() => onContentView(tag.id)}
                  initial={{ x: -20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.8 + (index * 0.1) }}
                  whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
                >
                  <span className="topic">{tag.name}</span>
                  <span className="topic-tooltip">{tag.desc}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>

          <motion.div
            className="navigation-buttons"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 1.2 }}
          >
            <motion.button
              onClick={() => onContentView()}
              whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="button-text">browse.content()</span>
              <span className="button-glow"></span>
            </motion.button>
          </motion.div>

          <SocialLinks />

          <div className="terminal-decoration">
            <div className="terminal-line"></div>
            <div className="terminal-dots">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>

          {/* Decorative elements */}
          <div className="decorative-elements">
            <div className="code-snippet left">
              <span className="code-line">const brain = new NeuralNetwork();</span>
              <span className="code-line">brain.train(data, {'{'}epochs: 1000{'}'});</span>
              <span className="code-line">const output = brain.predict(input);</span>
            </div>
            <div className="code-snippet right">
              <span className="code-line">function createIdeas() {'{'}</span>
              <span className="code-line">  return human.collaborate(machine);</span>
              <span className="code-line">{'}'}</span>
            </div>
          </div>
        </motion.div>
      </section>
    </div>
  );
};

export default HeroSection; 