import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import '../styles/HeroSection.css';

interface HeroSectionProps {
  onContentView: (category?: string) => void;
}

const HeroSection = ({ onContentView }: HeroSectionProps) => {
  const [typedText, setTypedText] = useState('');
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
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

  const handleTagHover = (id: string) => {
    setSelectedTag(id);
  };

  const handleTagLeave = () => {
    setSelectedTag(null);
  };

  const [showContent, setShowContent] = useState(false);

  return (
    <motion.div 
      className="dark-panel"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <motion.section 
        className="hero-section"
        initial={{ y: 20 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <motion.div 
          className="hero-content"
          layout
        >
          <motion.pre 
            className="ascii-art"
            initial={{ scale: 0.95, opacity: 0, y: 30 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            transition={{ 
              duration: 1.2, 
              delay: 0.4,
              type: "spring",
              stiffness: 80,
              damping: 15
            }}
          >
            {asciiArt}
          </motion.pre>

          <motion.div 
            className="terminal"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <div className="fixed-width-container">
              <p className="typed-text">{typedText}</p>
              <span className="cursor" />
            </div>
          </motion.div>
          
          <motion.div 
            className="topics-container"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.8 }}
          >
            <div className="topics">
              {tags.map((tag, index) => (
                <motion.div
                  key={tag.id}
                  className={`topic-wrapper ${selectedTag === tag.id ? 'active' : ''}`}
                  whileHover={{ y: -2, scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ 
                    duration: 0.3,
                    delay: 1 + (index * 0.1),
                    type: "spring",
                    stiffness: 300
                  }}
                  onClick={() => onContentView(tag.id)}
                >
                  <span className="topic">{tag.name}</span>
                  <AnimatePresence>
                    {selectedTag === tag.id && (
                      <motion.div 
                        className="topic-description"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                      >
                        <span className="description-text">{tag.desc}</span>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
            </div>
          </motion.div>

          <motion.div 
            className="navigation-buttons"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 1.2 }}
          >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => document.getElementById('journey')?.scrollIntoView({ behavior: 'smooth' })}
            >
              explore.brain()
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onContentView()}
            >
              browse.content()
            </motion.button>
          </motion.div>

          <motion.div 
            className="scroll-indicator"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ 
              duration: 0.5,
              delay: 1.4,
              repeat: Infinity,
              repeatType: "reverse"
            }}
            onClick={() => document.getElementById('journey')?.scrollIntoView({ behavior: 'smooth' })}
          >
            <span className="scroll-text">scroll.down()</span>
            <pre className="scroll-arrow">▼</pre>
          </motion.div>
        </motion.div>
      </motion.section>
      
      <section className="journey-section">
        {/* journey content */}
      </section>
    </motion.div>
  );
};

export default HeroSection; 