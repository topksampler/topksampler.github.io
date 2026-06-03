import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Plugin, ViteDevServer } from 'vite'

interface WriterSideNote {
  id: string;
  anchor: string;
  note: string;
  side?: 'left' | 'right';
}

interface WriterPost {
  id: string;
  title: string;
  date: string;
  content: string;
  sideNotes?: WriterSideNote[];
}

const postsDir = path.resolve(__dirname, 'src/content/posts');
const writerPostPath = (id: string) => path.join(postsDir, `${id}.json`);
const slugPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

const sendJson = (res: ServerResponse, status: number, data: unknown) => {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(data));
};

const readJsonBody = async (req: IncomingMessage) => {
  const chunks: Buffer[] = [];

  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }

  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
};

const normalizePost = (input: WriterPost): WriterPost => ({
  id: input.id,
  title: input.title,
  date: input.date,
  content: input.content,
  sideNotes: (input.sideNotes || []).map((note, index) => ({
    id: String(index + 1),
    anchor: note.anchor,
    note: note.note,
    side: note.side === 'left' ? 'left' : 'right'
  }))
});

const readPost = async (id: string) => {
  const raw = await readFile(writerPostPath(id), 'utf8');
  return JSON.parse(raw) as WriterPost;
};

const listPosts = async () => {
  const files = await readdir(postsDir);
  const posts = await Promise.all(
    files
      .filter((file) => file.endsWith('.json'))
      .map(async (file) => {
        const id = file.replace(/\.json$/, '');
        const post = await readPost(id);

        return {
          id: post.id || id,
          title: post.title || id,
          date: post.date || '',
          fileName: file
        };
      })
  );

  return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
};

const writerApiPlugin = (): Plugin => ({
  name: 'caffeine-writer-api',
  configureServer(server: ViteDevServer) {
    server.middlewares.use(async (req: IncomingMessage, res: ServerResponse, next: () => void) => {
      const requestUrl = new URL(req.url || '/', 'http://localhost');

      if (!requestUrl.pathname.startsWith('/api/writer/posts')) {
        next();
        return;
      }

      try {
        const id = decodeURIComponent(requestUrl.pathname.replace('/api/writer/posts/', ''));

        if (req.method === 'GET' && requestUrl.pathname === '/api/writer/posts') {
          sendJson(res, 200, { posts: await listPosts() });
          return;
        }

        if (req.method === 'GET' && id && slugPattern.test(id)) {
          sendJson(res, 200, { post: await readPost(id) });
          return;
        }

        if ((req.method === 'POST' || req.method === 'PUT') && requestUrl.pathname === '/api/writer/posts') {
          const post = normalizePost(await readJsonBody(req));

          if (!slugPattern.test(post.id)) {
            sendJson(res, 400, { error: 'Post id must be a lowercase url slug.' });
            return;
          }

          if (!post.title.trim() || !post.date || !post.content.trim()) {
            sendJson(res, 400, { error: 'Title, date, and content are required.' });
            return;
          }

          await writeFile(writerPostPath(post.id), `${JSON.stringify(post, null, 2)}\n`, 'utf8');
          sendJson(res, 200, {
            post,
            fileName: `${post.id}.json`,
            path: `src/content/posts/${post.id}.json`,
            url: `/post/${post.id}`
          });
          return;
        }

        sendJson(res, 404, { error: 'Writer API route not found.' });
      } catch (error) {
        sendJson(res, 500, { error: error instanceof Error ? error.message : 'Writer API failed.' });
      }
    });
  }
});

// https://vite.dev/config/
export default defineConfig({
  base: '/',
  plugins: [react(), writerApiPlugin()],
  assetsInclude: ['**/*.md'],
  build: {
    rollupOptions: {
    }
  }
});
