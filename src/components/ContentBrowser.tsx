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

const cardVariants = {
  initial: {
    opacity: 0,
    y: 20
  },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 30
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

const conceptsAscii = `
┌──────────────┐
│ ▄▄▄▄▄▄▄▄▄▄▄ │
│ █ AI/ML    █ │
│ █ Concepts █ │
│ ▀▀▀▀▀▀▀▀▀▀▀ │
└──────────────┘`;

const tutorialsAscii = `
┌──────────────┐
│ ┌─────┐     │
│ │CODE│ >>   │
│ └┬─┬┬┘      │
│ │ │ └──┼──┐ │
│ │ └────┴──┘ │
└──────────────┘`;

const projectsAscii = `
┌──────────────┐
│ ┌────────┐  │
│ │PROJECTS│  │
│ └───┬────┘  │
│ ┌───┴────┐  │
│ │ 🔨 ⚙️  📊 │  │
└──────────────┘`;

const thoughtsAscii = `
┌──────────────┐
│  ⚡️HUMAN⚡️  │
│  ┌──AI──┐   │
│  │SPARK│    │
│  └─────┘++  │
└──────────────┘`;

const MatrixRain = () => {
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

const TerminalHeader = ({ title }: { title: string }) => (
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
  const [selectedArticle, setSelectedArticle] = useState<ContentNode | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(category || initialCategory);
  const [content, setContent] = useState<ContentNode[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    document.documentElement.classList.add('content-active');
    document.body.classList.add('content-active');
    
    return () => {
      document.documentElement.classList.remove('content-active');
      document.body.classList.remove('content-active');
    };
  }, []);

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
    if (!selectedCategory && !articleId) {
      navigate('/content/concepts');
      setSelectedCategory('concepts');
    }
  }, [selectedCategory, articleId, navigate]);

  const handleCardClick = (article: ContentNode) => {
    setSelectedArticle(article);
    navigate(`/content/${article.category}/${article.id}`);
  };

  const handleBack = () => {
    if (location.pathname.includes('/content/')) {
      const pathParts = location.pathname.split('/');
      if (pathParts.length > 3) {
        navigate(`/content/${pathParts[2]}`);
        setSelectedArticle(null);
      } else {
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

  const renderArticle = (article: ContentNode) => {
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
              variants={cardVariants}
              initial="initial"
              animate="visible"
              exit="exit"
            >
              {filteredContent.map((article, index) => (
                <motion.div
                  key={`${article.id}-card`}
                  className={`content-card ${article.category}`}
                  variants={cardVariants}
                  initial="initial"
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