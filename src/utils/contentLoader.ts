export interface SideNote {
  id: string;
  anchor: string;
  note: string;
}

export interface Post {
  id: string;
  title: string;
  date: string;
  content: string;
  sideNotes?: SideNote[];
}

export function loadPosts(): Post[] {
  // Use Vite's glob import to load all JSON files from the posts directory
  const modules = import.meta.glob('../content/posts/*.json', { eager: true });

  const posts = Object.values(modules).map((module: any) => {
    return module.default || module;
  });

  // Sort posts by date (newest first)
  return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}