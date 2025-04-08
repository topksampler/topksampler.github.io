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

export async function watchContent(callback: (content: ContentData[]) => void) {
  try {
    // Get module paths using import.meta.glob, but don't load content eagerly
    const modules = import.meta.glob<false, string>('/public/content/**/*.md');

    const contentPromises = Object.entries(modules).map(async ([path]) => {
      try {
        // Construct the correct fetch path (remove /public prefix)
        const fetchPath = path.replace(/^\/public/, '');
        const response = await fetch(fetchPath);
        if (!response.ok) {
          throw new Error(`Failed to fetch ${fetchPath}: ${response.statusText}`);
        }
        const rawContentWithFrontmatter = await response.text();

        // Use gray-matter to parse
        const { data: frontmatter, content: rawContent } = matter(rawContentWithFrontmatter);

        // Extract filename as id if not present in frontmatter
        const id = frontmatter.id || path.split('/').pop()?.replace('.md', '') || 'unknown';

        // Basic category extraction from path if not in frontmatter
        const pathParts = path.split('/');
        const category = frontmatter.category || (pathParts.length > 3 ? pathParts[pathParts.length - 2] : 'unknown'); // Adjusted index for /public/content/category/file.md

        // Split content into sections based on '## '
        const sectionsRaw = rawContent.split(/\n(?=##\s)/);
        const intro = sectionsRaw.length > 0 && !sectionsRaw[0].startsWith('## ') ? sectionsRaw.shift()?.trim() || '' : '';

        const sections: Section[] = sectionsRaw.map(sectionText => {
          const lines = sectionText.trim().split('\n');
          const title = lines[0]?.replace(/^##\s+/, '').trim() || 'Untitled Section';
          const contentBody = lines.slice(1).join('\n').trim();

          const codeMatch = contentBody.match(/```(?:\w+)?\n([\s\S]*?)```/);
          const code = codeMatch ? codeMatch[1].trim() : undefined;
          const content = codeMatch ? contentBody.replace(codeMatch[0], '').trim() : contentBody;

          return { title, content, code };
        }).filter(s => s.title || s.content);

        let conclusion: string | undefined;
        if (sections.length > 0 && sections[sections.length - 1].title.toLowerCase().includes('conclusion')) {
          conclusion = sections.pop()?.content;
        }

        return {
          ...frontmatter,
          id,
          category,
          content: {
            intro,
            sections,
            conclusion,
          },
        } as ContentData;
      } catch (error) {
        console.error(`Error processing markdown file ${path}:`, error);
        return null;
      }
    });

    const contentFiles = await Promise.all(contentPromises);

    // Filter out any null results and sort by date (if date exists)
    const validContent = contentFiles
      .filter((content): content is ContentData => content !== null)
      .sort((a, b) => {
        try {
          // Ensure dates are valid before comparing
          const dateA = a.date ? new Date(a.date).getTime() : 0;
          const dateB = b.date ? new Date(b.date).getTime() : 0;
          if (isNaN(dateA) || isNaN(dateB)) return 0; // Handle invalid dates
          return dateB - dateA;
        } catch (_e) {
          console.error("Error during date sorting:", _e);
          return 0; // Fallback if date parsing fails
        }
      });

    callback(validContent);
  } catch (error) {
    console.error('Error watching content:', error);
    callback([]);
  }
}

export type { ContentData };