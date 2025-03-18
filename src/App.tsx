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

  // Navigation handled through React Router

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
            />
          } />
          <Route path="/content/:category" element={
            <ContentBrowser 
              initialCategory={location.pathname.split('/')[2] || null}
            />
          } />
          <Route path="/content/:category/:articleId" element={
            <ContentBrowser 
              initialCategory={location.pathname.split('/')[2] || null}
              articleId={location.pathname.split('/')[3] || null}
            />
          } />
        </Routes>
      </main>
    </div>
  )
}

export default App
