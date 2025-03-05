import { useEffect, useState } from 'react';
import '../styles/HeroSection.css';

const HeroSection = () => {
  const [typedText, setTypedText] = useState('');
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const fullText = '> exploring the intersection of human creativity and artificial intelligence';
  const asciiArt = `
  ╔══════════════════╗
  ║   CaffeineBrain  ║
  ║      [ AI ]      ║
  ╚══════════════════╝
  `;

  const tags = [
    { id: 'mlp', name: 'MLPs', desc: 'Multi-Layer Perceptrons: The Neural Building Blocks' },
    { id: 'transformer', name: 'Transformers', desc: 'Attention Is All You Need' },
    { id: 'neural', name: 'Neural Networks', desc: 'Biological Intelligence Inspired' },
    { id: 'deep', name: 'Deep Learning', desc: 'Going Deeper Into The Unknown' },
    { id: 'ethics', name: 'AI Ethics', desc: 'Responsible Innovation' }
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

  return (
    <section className="hero-section">
      <div className="hero-content">
        <pre className="ascii-art">{asciiArt}</pre>
        <div className="terminal">
          <p className="typed-text">{typedText}<span className="cursor">_</span></p>
        </div>
        
        <div className="topics-container">
          <div className="topics">
            {tags.map(tag => (
              <div
                key={tag.id}
                className={`topic-wrapper ${selectedTag === tag.id ? 'active' : ''}`}
                onMouseEnter={() => handleTagHover(tag.id)}
                onMouseLeave={handleTagLeave}
              >
                <span className="topic">{tag.name}</span>
                <div className="topic-description">
                  <span className="description-text">{tag.desc}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="navigation-buttons">
          <button onClick={() => document.getElementById('journey')?.scrollIntoView({ behavior: 'smooth' })}>
            start.journey()
          </button>
          <button onClick={() => document.getElementById('blog')?.scrollIntoView({ behavior: 'smooth' })}>
            read.blog()
          </button>
        </div>

        <div className="scroll-indicator">
          <span className="scroll-text">scroll.down()</span>
          <pre className="scroll-arrow">
            {`
   ┃
   ┃
   ▼
            `}
          </pre>
        </div>
      </div>
    </section>
  );
};

export default HeroSection; 