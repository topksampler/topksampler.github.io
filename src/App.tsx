import { useState, useEffect } from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import './App.css';
import HomePage from './components/HomePage';
import ReadingPage from './components/ReadingPage';
import { loadPosts, Post } from './utils/contentLoader';

function App() {
  const navigate = useNavigate();
  // Initialize posts from the loader
  const [posts] = useState<Post[]>(loadPosts());
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);

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

  const handleNavigate = (postId: string) => {
    const post = posts.find(p => p.id === postId);
    if (post) {
      setSelectedPost(post);
      navigate(`/post/${postId}`);
      window.scrollTo(0, 0);
    }
  };

  const getAdjacentPosts = (currentId: string) => {
    const currentIndex = posts.findIndex(p => p.id === currentId);
    return {
      prev: currentIndex < posts.length - 1 ? posts[currentIndex + 1] : null,
      next: currentIndex > 0 ? posts[currentIndex - 1] : null
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
            />
          }
        />
        <Route
          path="/post/:postId"
          element={
            selectedPost ? (
              <ReadingPage
                post={selectedPost}
                prevPost={getAdjacentPosts(selectedPost.id).prev}
                nextPost={getAdjacentPosts(selectedPost.id).next}
                onBack={handleBack}
                onNavigate={handleNavigate}
              />
            ) : (
              <HomePage
                posts={posts}
                onPostClick={handlePostClick}
              />
            )
          }
        />
      </Routes>
    </div>
  );
}

export default App;
