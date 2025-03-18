import { useEffect, useState } from 'react';
import '../styles/HeroSection.css';

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
        <div className="hero-content">
          <pre className="ascii-art">{asciiArt}</pre>

          <div className="terminal">
            <div className="fixed-width-container">
              <p className="typed-text">{typedText}</p>
              <span className="cursor" />
            </div>
          </div>
          
          <div className="topics-container">
            <div className="topics">
              {tags.map((tag) => (
                <div
                  key={tag.id}
                  className="topic-wrapper"
                  onClick={() => onContentView(tag.id)}
                >
                  <span className="topic">{tag.name}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="navigation-buttons">
            <button onClick={() => onContentView()}>
              browse.content()
            </button>
          </div>
        </div>
      </section>
      
      <section className="journey-section">
        {/* journey content */}
      </section>
    </div>
  );
};

export default HeroSection; 