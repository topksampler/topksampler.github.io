export interface ContentData {
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