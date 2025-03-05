import { useEffect } from 'react'
import './App.css'
import HeroSection from './components/HeroSection'
import JourneySection from './components/JourneySection'
import ParticlesBackground from './components/ParticlesBackground'

function App() {
  useEffect(() => {
    // Smooth scroll behavior
    document.documentElement.style.scrollBehavior = 'smooth'
  }, [])

  return (
    <div className="app">
      <div className="noise" />
      <ParticlesBackground />
      <main className="main-content">
        <HeroSection />
        <JourneySection />
      </main>
    </div>
  )
}

export default App
