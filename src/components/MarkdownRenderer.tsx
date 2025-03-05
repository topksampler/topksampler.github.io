import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import '../styles/MarkdownRenderer.css';

interface MarkdownRendererProps {
  content: string;
  onLinkClick?: (href: string) => void;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, onLinkClick }) => {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        rehypePlugins={[rehypeHighlight, rehypeRaw]}
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ node, ...props }) => <h1 className="md-h1" {...props} />,
          h2: ({ node, ...props }) => <h2 className="md-h2" {...props} />,
          h3: ({ node, ...props }) => <h3 className="md-h3" {...props} />,
          p: ({ node, ...props }) => <p className="md-p" {...props} />,
          pre: ({ node, ...props }) => <pre className="md-pre" {...props} />,
          code: ({ node, className, children, ...props }) => {
            const isInline = !className;
            return (
              <code
                className={`md-code ${isInline ? 'inline' : ''} ${className || ''}`}
                {...props}
              >
                {children}
              </code>
            );
          },
          ul: ({ node, ...props }) => <ul className="md-ul" {...props} />,
          ol: ({ node, ...props }) => <ol className="md-ol" {...props} />,
          li: ({ node, ...props }) => <li className="md-li" {...props} />,
          blockquote: ({ node, ...props }) => <blockquote className="md-blockquote" {...props} />,
          a: ({ node, ...props }) => (
            <a
              {...props}
              onClick={(e) => {
                e.preventDefault();
                if (onLinkClick && props.href) {
                  onLinkClick(props.href);
                }
              }}
              style={{ cursor: 'pointer' }}
            />
          ),
          img: ({ node, ...props }) => <img className="md-img" {...props} />,
          hr: ({ node, ...props }) => <hr className="md-hr" {...props} />,
          table: ({ node, ...props }) => <table className="md-table" {...props} />,
          th: ({ node, ...props }) => <th className="md-th" {...props} />,
          td: ({ node, ...props }) => <td className="md-td" {...props} />
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}; 