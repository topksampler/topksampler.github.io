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
    if (location.pathname.includes('/content/')) {
      navigate('/content');
    } else {
      navigate('/');
    }
  }

  return (
    <div className="app">
      <div className="noise" />
      <main className="main-content">
        <Routes>
          <Route path="/" element={
            <>
              <HeroSection onContentView={handleContentView} />
              <JourneySection onContentView={handleContentView} />
            </>
          } />
          <Route path="/content" element={
            <ContentBrowser 
              initialCategory={null}
              onBack={handleBack}
            />
          } />
          <Route path="/content/:category" element={
            <ContentBrowser 
              initialCategory={location.pathname.split('/').pop() || null}
              onBack={handleBack}
            />
          } />
        </Routes>
      </main>
    </div>
  )
}

export default App
