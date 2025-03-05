declare module 'react-markdown' {
  import { ReactNode } from 'react';

  interface ReactMarkdownProps {
    children: string;
    className?: string;
    rehypePlugins?: any[];
    remarkPlugins?: any[];
    components?: {
      [key: string]: React.ComponentType<any>;
    };
  }

  export default function ReactMarkdown(props: ReactMarkdownProps): ReactNode;
}

declare module 'rehype-raw' {
  const rehypeRaw: any;
  export default rehypeRaw;
}

declare module 'remark-gfm' {
  const remarkGfm: any;
  export default remarkGfm;
} 