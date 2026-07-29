import { motion } from 'framer-motion';
import '../styles/HomePage.css';
import type { Post, PostCategory } from '../utils/contentLoader';
import SocialLinks from './SocialLinks';

interface HomePageProps {
    posts: Post[];
    onPostClick: (postId: string) => void;
    onAboutClick: () => void;
}

const HomePage = ({ posts, onPostClick, onAboutClick }: HomePageProps) => {
    const asciiArt = `
   ██████╗ █████╗ ███████╗███████╗███████╗██╗███╗   ██╗███████╗
  ██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔════╝
  ██║     ███████║█████╗  █████╗  █████╗  ██║██╔██╗ ██║█████╗  
  ██║     ██╔══██║██╔══╝  ██╔══╝  ██╔══╝  ██║██║╚██╗██║██╔══╝  
  ╚██████╗██║  ██║██║     ██║     ███████╗██║██║ ╚████║███████╗
   ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝
                                                          
  ██████╗ ██████╗  █████╗ ██╗███╗   ██╗                        
  ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                        
  ██████╔╝██████╔╝███████║██║██╔██╗ ██║                        
  ██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║                        
  ██████╔╝██║  ██║██║  ██║██║██║ ╚████║                        
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝                        
`;

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.15,
                delayChildren: 0.3
            }
        }
    };

    const postVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: {
            opacity: 1,
            y: 0,
            transition: {
                type: "spring",
                stiffness: 100,
                damping: 15
            }
        }
    };

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
        });
    };

    const ledgers: Array<{
        category: PostCategory;
        index: string;
        label: string;
        posts: Post[];
    }> = [
        {
            category: 'philosophy',
            index: '01',
            label: 'philosophy',
            posts: posts.filter(post => post.category === 'philosophy')
        },
        {
            category: 'technical',
            index: '02',
            label: 'technical',
            posts: posts.filter(post => post.category === 'technical')
        }
    ];

    return (
        <div className="home-page">
            <div className="noise-overlay" />

            <motion.header
                className="header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
            >
                <motion.pre
                    className="ascii-logo"
                    role="img"
                    aria-label="Caffeine Brain"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.8, delay: 0.2 }}
                >
                    {asciiArt}
                </motion.pre>

                <SocialLinks />

                <motion.button
                    type="button"
                    className="about-link"
                    onClick={onAboutClick}
                    whileHover={{ x: 3 }}
                    whileTap={{ scale: 0.97 }}
                >
                    03 / about
                </motion.button>

                <div className="divider" />
            </motion.header>

            <motion.main
                className="ledgers"
                variants={containerVariants}
                initial="hidden"
                animate="visible"
            >
                {ledgers.map(ledger => (
                    <section
                        key={ledger.category}
                        className={`ledger ledger--${ledger.category}`}
                        aria-labelledby={`${ledger.category}-heading`}
                    >
                        <div className="ledger-header">
                            <h2 id={`${ledger.category}-heading`} className="ledger-title">
                                <span className="ledger-index">{ledger.index} /</span>
                                {ledger.label}
                            </h2>
                            <span
                                className="ledger-count"
                                aria-label={`${ledger.posts.length} ${ledger.posts.length === 1 ? 'post' : 'posts'}`}
                            >
                                {String(ledger.posts.length).padStart(2, '0')}
                            </span>
                        </div>

                        <motion.ul className="ledger-posts" variants={containerVariants}>
                            {ledger.posts.map((post, index) => (
                                <motion.li
                                    key={post.id}
                                    className="post-entry"
                                    variants={postVariants}
                                >
                                    <motion.button
                                        type="button"
                                        className="post-item"
                                        whileHover={{ x: 6 }}
                                        whileTap={{ scale: 0.99 }}
                                        onClick={() => onPostClick(post.id)}
                                    >
                                        <span className="post-date">{formatDate(post.date)}</span>
                                        <span className="post-title">{post.title}</span>
                                        {index === 0 && <span className="latest-badge">latest</span>}
                                    </motion.button>
                                </motion.li>
                            ))}
                        </motion.ul>
                    </section>
                ))}
            </motion.main>

            <footer className="footer">
                <div className="divider" />
                <span className="footer-text">where the map meets the territory</span>
            </footer>
        </div>
    );
};

export default HomePage;
