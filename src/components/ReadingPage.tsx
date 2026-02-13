import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import '../styles/ReadingPage.css';

interface Post {
    id: string;
    title: string;
    date: string;
    content: string;
    sideNotes?: SideNote[];
}

interface SideNote {
    id: string;
    anchor: string;
    note: string;
    side?: 'left' | 'right';
}

interface ReadingPageProps {
    post: Post;
    prevPost?: { id: string; title: string } | null;
    nextPost?: { id: string; title: string } | null;
    onBack: () => void;
    onNavigate: (postId: string) => void;
}

// Split content into sections and match notes to sections
const useSectionsWithNotes = (content: string, sideNotes?: SideNote[]) => {
    return useMemo(() => {
        // Split content by ## headings
        const sections = content.split(/(?=^## )/gm).filter(s => s.trim());

        return sections.map((section, index) => {
            // Find notes that belong to this section (anchor text appears in section)
            const matchingNotes = sideNotes?.filter(note =>
                section.toLowerCase().includes(note.anchor.toLowerCase())
            ) || [];

            return {
                id: index,
                content: section,
                notes: matchingNotes
            };
        });
    }, [content, sideNotes]);
};

const ReadingPage: React.FC<ReadingPageProps> = ({
    post,
    prevPost,
    nextPost,
    onBack,
    onNavigate
}) => {
    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            month: 'long',
            day: 'numeric',
            year: 'numeric'
        });
    };

    // Same ASCII logo as homepage
    const miniLogo = `   ██████╗ █████╗ ███████╗███████╗███████╗██╗███╗   ██╗███████╗
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

    const sectionsWithNotes = useSectionsWithNotes(post.content, post.sideNotes);

    const markdownComponents = {
        code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
                <div className="code-block">
                    <div className="code-header">
                        <span className="code-lang">{match[1]}</span>
                    </div>
                    <SyntaxHighlighter
                        style={atomDark}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                    >
                        {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                </div>
            ) : (
                <code className={className} {...props}>
                    {children}
                </code>
            );
        },
        blockquote({ children }: any) {
            return (
                <blockquote className="styled-quote">
                    {children}
                </blockquote>
            );
        }
    };

    return (
        <div className="reading-page">
            <div className="noise-overlay" />

            <motion.header
                className="reading-header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
            >
                <motion.button
                    className="back-btn"
                    onClick={onBack}
                    whileHover={{ x: -4 }}
                    whileTap={{ scale: 0.95 }}
                >
                    ← back
                </motion.button>
                <pre className="mini-logo" onClick={onBack}>{miniLogo}</pre>
                <span className="reading-date">{formatDate(post.date)}</span>
            </motion.header>

            <div className="reading-layout">
                <motion.article
                    className="reading-content"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                >
                    <div className="title-divider" />
                    <h1 className="reading-title">{post.title}</h1>
                    <div className="title-divider" />

                    {/* Desktop: render sections with side notes absolutely positioned */}
                    <div className="markdown-body desktop-content">
                        {sectionsWithNotes.map((section) => {
                            const leftNotes = section.notes.filter(n => n.side === 'left');
                            const rightNotes = section.notes.filter(n => n.side === 'right' || !n.side);

                            return (
                                <div key={section.id} className="content-section-desktop">
                                    {/* Left notes - absolutely positioned */}
                                    {leftNotes.length > 0 && (
                                        <div className="side-notes-absolute left">
                                            {leftNotes.map((note) => (
                                                <div key={note.id} className="side-note">
                                                    <p>{note.note}</p>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    {/* Main content */}
                                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                                        {section.content}
                                    </ReactMarkdown>

                                    {/* Right notes - absolutely positioned */}
                                    {rightNotes.length > 0 && (
                                        <div className="side-notes-absolute right">
                                            {rightNotes.map((note) => (
                                                <div key={note.id} className="side-note">
                                                    <p>{note.note}</p>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>

                    {/* Mobile: render sections with notes stacked after each */}
                    <div className="markdown-body mobile-content">
                        {sectionsWithNotes.map((section) => (
                            <div key={section.id} className="content-section">
                                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                                    {section.content}
                                </ReactMarkdown>

                                {section.notes.length > 0 && (
                                    <div className="section-notes">
                                        {section.notes.map((note) => (
                                            <div key={note.id} className="section-note">
                                                <span className="section-note-marker">◆</span>
                                                <p>{note.note}</p>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </motion.article>
            </div>

            <motion.footer
                className="reading-footer"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
            >
                <div className="nav-divider" />
                <nav className="post-navigation">
                    {prevPost ? (
                        <motion.button
                            className="nav-btn prev"
                            onClick={() => onNavigate(prevPost.id)}
                            whileHover={{ x: -4 }}
                        >
                            <span className="nav-arrow">←</span>
                            <span className="nav-title">{prevPost.title}</span>
                        </motion.button>
                    ) : <div />}

                    {nextPost ? (
                        <motion.button
                            className="nav-btn next"
                            onClick={() => onNavigate(nextPost.id)}
                            whileHover={{ x: 4 }}
                        >
                            <span className="nav-title">{nextPost.title}</span>
                            <span className="nav-arrow">→</span>
                        </motion.button>
                    ) : <div />}
                </nav>
            </motion.footer>
        </div>
    );
};

export default ReadingPage;
