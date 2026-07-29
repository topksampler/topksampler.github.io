import { motion } from 'framer-motion';
import '../styles/AboutPage.css';

interface AboutPageProps {
    onBack: () => void;
}

const AboutPage = ({ onBack }: AboutPageProps) => {
    return (
        <div className="about-page">
            <div className="noise-overlay" />

            <motion.header
                className="about-header"
                initial={{ opacity: 0, y: -16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
            >
                <motion.button
                    type="button"
                    className="about-back"
                    onClick={onBack}
                    whileHover={{ x: -4 }}
                    whileTap={{ scale: 0.97 }}
                >
                    ← index
                </motion.button>

                <button type="button" className="about-wordmark" onClick={onBack}>
                    caffeine brain
                </button>

                <span className="about-route">03 / about</span>
            </motion.header>

            <motion.main
                className="about-layout"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.55, delay: 0.15 }}
            >
                <section className="about-identity" aria-labelledby="about-name">
                    <span className="about-kicker">identity / author</span>
                    <h1 id="about-name" className="about-name">
                        Lalithnarayan
                        <span>Chandrashekar</span>
                    </h1>
                    <figure className="about-portrait">
                        <img
                            src="/images/about/lalithnarayan-chandrashekar.jpg"
                            alt="Lalithnarayan Chandrashekar standing among rocks beside a mountain stream"
                        />
                        <figcaption>
                            <span>field note / away from the desk</span>
                            <span>01</span>
                        </figcaption>
                    </figure>
                    <span className="about-coordinate">ml systems × inner life</span>
                </section>

                <section className="about-copy" aria-label="About Lalithnarayan Chandrashekar">
                    <p className="about-lede">
                        I build ML systems close to the metal - and write about the life
                        happening above it.
                    </p>

                    <div className="about-rule" />

                    <p>
                        At AMD, I work on fast LLM inference. My public engineering work crosses
                        vLLM, PyTorch, Transformers, ZenDNN, and emerging CPU serving paths.
                    </p>

                    <p>
                        This site is my exploration dump. Some threads begin in technical
                        systems; others begin in philosophy or inner life. The structure
                        matters - it gives each thought somewhere to live - but the boundaries are
                        temporary. At some point, the threads become relevant to one another.
                    </p>

                    <dl className="about-ledger">
                        <div>
                            <dt>work</dt>
                            <dd>ML systems / AMD</dd>
                        </div>
                        <div>
                            <dt>focus</dt>
                            <dd>LLM inference / CPU systems</dd>
                        </div>
                        <div>
                            <dt>writing</dt>
                            <dd>philosophy / technical</dd>
                        </div>
                    </dl>

                    <nav className="about-profiles" aria-label="Public profiles">
                        <a
                            href="https://github.com/amd-lalithnc"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            <span>engineering / in public</span>
                            <strong>amd-lalithnc ↗</strong>
                        </a>
                        <a
                            href="https://github.com/topksampler"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            <span>elsewhere / as topksampler</span>
                            <strong>topksampler ↗</strong>
                        </a>
                        <a
                            href="https://www.linkedin.com/in/lalithnarayan-c/"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            <span>linkedin</span>
                            <strong>lalithnarayan-c ↗</strong>
                        </a>
                    </nav>
                </section>
            </motion.main>

            <footer className="about-footer">
                <span>where the map meets the territory</span>
            </footer>
        </div>
    );
};

export default AboutPage;
