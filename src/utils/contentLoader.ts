import { ContentData } from '../types';
import matter from 'gray-matter';
import { Buffer } from 'buffer';

// Polyfill Buffer for browser environment
globalThis.Buffer = Buffer;

interface Section {
  title: string;
  content: string;
  code?: string;
}

async function loadMarkdownFile(filePath: string): Promise<ContentData | null> {
  try {
    // Remove the leading slash as Vite will resolve relative to the project root
    const cleanPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
    const response = await fetch(cleanPath);
    const text = await response.text();
    const { data, content } = matter(text);
    
    // Parse the frontmatter
    const {
      id,
      title,
      category,
      description,
      date,
      readingTime,
      ascii
    } = data as Omit<ContentData, 'content'>;

    // Split content into sections
    const sections = content.split('\n## ').map(section => {
      if (!section.includes('\n')) return null;
      const [title, ...contentParts] = section.split('\n');
      const sectionContent = contentParts.join('\n').trim();
      
      // Check if there's a code block
      const codeMatch = sectionContent.match(/```[\w]*\n([\s\S]*?)```/);
      const code = codeMatch ? codeMatch[1].trim() : undefined;
      const textContent = sectionContent.replace(/```[\w]*\n[\s\S]*?```/g, '').trim();

      return {
        title: title.replace('## ', '').trim(),
        content: textContent,
        code
      } as Section;
    }).filter((section): section is Section => section !== null);

    // First section is the intro
    const intro = sections.shift()?.content || '';

    // Last section might be conclusion
    let conclusion: string | undefined;
    if (sections.length > 0 && sections[sections.length - 1].title.toLowerCase().includes('conclusion')) {
      conclusion = sections.pop()?.content;
    }

    return {
      id,
      title,
      category,
      description,
      date,
      readingTime,
      ascii,
      content: {
        intro,
        sections,
        conclusion
      }
    };
  } catch (error) {
    console.error('Error loading markdown file:', error);
    return null;
  }
}

export async function watchContent(callback: (content: ContentData[]) => void) {
  try {
    const contentFiles = await Promise.all(
      Object.entries({
        ...import.meta.glob('../../content/concepts/*.md'),
        ...import.meta.glob('../../content/tutorials/*.md'),
        ...import.meta.glob('../../content/projects/*.md'),
        ...import.meta.glob('../../content/thoughts/*.md')
      }).map(async ([path, _]) => {
        const content = await loadMarkdownFile(path);
        return content;
      })
    );

    // Filter out any null results and sort by date
    const validContent = contentFiles
      .filter((content: ContentData | null): content is ContentData => content !== null)
      .sort((a: ContentData, b: ContentData) => new Date(b.date).getTime() - new Date(a.date).getTime());

    callback(validContent);
  } catch (error) {
    console.error('Error watching content:', error);
    callback([]);
  }
}

export type { ContentData }; 