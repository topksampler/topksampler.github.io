import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import './App.css'
import HeroSection from './components/HeroSection'
import JourneySection from './components/JourneySection'
import ContentBrowser from './components/ContentBrowser'

function App() {
  const navigate = useNavigate();
  const location = useLocation();

  const handleContentView = (category?: string) => {
    if (category) {
      navigate(`/content/${category}`);
    } else {
      navigate('/content');
    }
  }

  const handleBack = () => {
    const pathParts = location.pathname.split('/');
    
    // If we're in an article view (e.g., /content/category/article)
    if (pathParts.length > 3) {
      navigate(`/content/${pathParts[2]}`); // Go back to category view
    }
    // If we're in a category view (e.g., /content/category)
    else if (pathParts.length === 3) {
      navigate('/content'); // Go back to all content
    }
    // If we're in the main content view
    else if (location.pathname === '/content') {
      navigate('/'); // Go back to home
    }
  }

  return (
    <div className="app">
      <div className="noise" />
      <main className="main-content">
        <Routes>
          <Route path="/" element={
            <div className="content-container">
              <HeroSection onContentView={handleContentView} />
              <JourneySection onContentView={handleContentView} />
            </div>
          } />
          <Route path="/content" element={
            <ContentBrowser 
              initialCategory={null}
              onBack={handleBack}
            />
          } />
          <Route path="/content/:category" element={
            <ContentBrowser 
              initialCategory={location.pathname.split('/')[2] || null}
              onBack={handleBack}
            />
          } />
          <Route path="/content/:category/:articleId" element={
            <ContentBrowser 
              initialCategory={location.pathname.split('/')[2] || null}
              articleId={location.pathname.split('/')[3] || null}
              onBack={handleBack}
            />
          } />
        </Routes>
      </main>
    </div>
  )
}

export default App
