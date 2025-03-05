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
            explore.brain()
          </button>
          <button onClick={() => document.getElementById('blog')?.scrollIntoView({ behavior: 'smooth' })}>
            browse.content()
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