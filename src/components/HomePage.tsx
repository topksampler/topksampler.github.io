import { motion } from 'framer-motion';
import '../styles/HomePage.css';
import SocialLinks from './SocialLinks';

interface Post {
    id: string;
    title: string;
    date: string;
    preview?: string;
}

interface HomePageProps {
    posts: Post[];
    onPostClick: (postId: string) => void;
}

const HomePage = ({ posts, onPostClick }: HomePageProps) => {
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
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.8, delay: 0.2 }}
                >
                    {asciiArt}
                </motion.pre>

                <SocialLinks />

                <div className="divider" />
            </motion.header>

            <motion.main
                className="posts-container"
                variants={containerVariants}
                initial="hidden"
                animate="visible"
            >
                {posts.map((post, index) => (
                    <motion.article
                        key={post.id}
                        className="post-item"
                        variants={postVariants}
                        whileHover={{ x: 8 }}
                        onClick={() => onPostClick(post.id)}
                    >
                        <span className="post-date">{formatDate(post.date)}</span>
                        <h2 className="post-title">{post.title}</h2>
                        {index === 0 && <span className="latest-badge">latest</span>}
                    </motion.article>
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
