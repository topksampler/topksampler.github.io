import { useState, useEffect } from 'react';
import { Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import './App.css';
import AboutPage from './components/AboutPage';
import HomePage from './components/HomePage';
import ReadingPage from './components/ReadingPage';
import { loadPosts, Post } from './utils/contentLoader';

function App() {
  const navigate = useNavigate();
  const location = useLocation();
  // Initialize posts from the loader
  const [posts] = useState<Post[]>(loadPosts());
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);

  const routePostId = location.pathname.startsWith('/post/')
    ? decodeURIComponent(location.pathname.replace('/post/', '').split('/')[0])
    : null;
  const activePost = routePostId
    ? posts.find(p => p.id === routePostId) || null
    : selectedPost;

  const handlePostClick = (postId: string) => {
    const post = posts.find(p => p.id === postId);
    if (post) {
      setSelectedPost(post);
      navigate(`/post/${postId}`);
    }
  };

  const handleBack = () => {
    setSelectedPost(null);
    navigate('/');
  };

  const handleAboutClick = () => {
    navigate('/about');
    window.scrollTo(0, 0);
  };

  const handleNavigate = (postId: string) => {
    const post = posts.find(p => p.id === postId);
    if (post) {
      setSelectedPost(post);
      navigate(`/post/${postId}`);
      window.scrollTo(0, 0);
    }
  };

  const getAdjacentPosts = (currentId: string) => {
    const currentPost = posts.find(p => p.id === currentId);

    if (!currentPost) {
      return { prev: null, next: null };
    }

    const categoryPosts = posts.filter(p => p.category === currentPost.category);
    const currentIndex = categoryPosts.findIndex(p => p.id === currentId);

    return {
      prev: currentIndex < categoryPosts.length - 1 ? categoryPosts[currentIndex + 1] : null,
      next: currentIndex > 0 ? categoryPosts[currentIndex - 1] : null
    };
  };

  // Handle direct URL navigation
  useEffect(() => {
    const hash = window.location.hash;
    if (hash.includes('/post/')) {
      const postId = hash.split('/post/')[1];
      const post = posts.find(p => p.id === postId);
      if (post) {
        setSelectedPost(post);
      }
    }
  }, [posts]);

  return (
    <div className="app">
      <Routes>
        <Route
          path="/"
          element={
            <HomePage
              posts={posts}
              onPostClick={handlePostClick}
              onAboutClick={handleAboutClick}
            />
          }
        />
        <Route
          path="/about"
          element={<AboutPage onBack={handleBack} />}
        />
        <Route
          path="/post/:postId"
          element={
            activePost ? (
              <ReadingPage
                post={activePost}
                prevPost={getAdjacentPosts(activePost.id).prev}
                nextPost={getAdjacentPosts(activePost.id).next}
                onBack={handleBack}
                onNavigate={handleNavigate}
              />
            ) : (
              <HomePage
                posts={posts}
                onPostClick={handlePostClick}
                onAboutClick={handleAboutClick}
              />
            )
          }
        />
      </Routes>
    </div>
  );
}

export default App;
