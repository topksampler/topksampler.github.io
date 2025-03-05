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
          ul: ({ node, className, ...props }) => {
            const isTaskList = className?.includes('contains-task-list');
            return (
              <ul 
                className={isTaskList ? 'md-task-list' : 'md-ul'}
                {...props}
              />
            );
          },
          ol: ({ node, ...props }) => <ol className="md-ol" {...props} />,
          li: ({ node, className, ...props }) => {
            const isTaskListItem = className?.includes('task-list-item');
            return (
              <li 
                className={isTaskListItem ? 'md-task-list-item' : 'md-li'}
                {...props}
              />
            );
          },
          blockquote: ({ node, ...props }) => {
            // Check if this is a callout by looking for special syntax
            const text = props.children?.[0]?.props?.children?.[0] || '';
            const calloutMatch = text.match(/^!(info|warning|error|success)\s/);
            
            if (calloutMatch) {
              const type = calloutMatch[1];
              // Remove the callout syntax from the text
              props.children[0].props.children[0] = text.replace(/^!(info|warning|error|success)\s/, '');
              return <blockquote className={`md-callout ${type}`} {...props} />;
            }
            
            // Check if this is a note
            if (text.startsWith('Note: ')) {
              props.children[0].props.children[0] = text.replace(/^Note:\s/, '');
              return <blockquote className="md-note" {...props} />;
            }
            
            return <blockquote className="md-blockquote" {...props} />;
          },
          a: ({ node, ...props }) => (
            <a
              className="md-a"
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
          td: ({ node, ...props }) => <td className="md-td" {...props} />,
          dl: ({ node, ...props }) => <dl className="md-dl" {...props} />,
          dt: ({ node, ...props }) => <dt className="md-dt" {...props} />,
          dd: ({ node, ...props }) => <dd className="md-dd" {...props} />,
          kbd: ({ node, ...props }) => <kbd className="md-kbd" {...props} />,
          sub: ({ node, ...props }) => <sub className="md-sub" {...props} />,
          sup: ({ node, ...props }) => <sup className="md-sup" {...props} />
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}; 