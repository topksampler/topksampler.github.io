#!/usr/bin/env node
/**
 * Creative Art Generator using Replicate API
 * Uses ideogram-ai/ideogram-v3-turbo for colorful, whacky splash art
 * 
 * Usage:
 *   npm run generate-art                    # Generate with random creative prompt
 *   npm run generate-art "custom prompt"    # Generate with custom prompt
 *   npm run generate-art "prompt" myart     # Custom prompt and output name
 */

import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

// Load env manually since we're in tools/
const envPath = path.join(projectRoot, '.env');
if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf-8');
    envContent.split('\n').forEach(line => {
        const [key, ...valueParts] = line.split('=');
        if (key && valueParts.length) {
            process.env[key.trim()] = valueParts.join('=').trim();
        }
    });
}

// Creative prompt templates for whacky colorful art
const CREATIVE_PROMPTS = [
    "Abstract explosion of vibrant neon colors, paint splashes, cosmic energy, highly detailed digital art, trending on artstation",
    "Psychedelic liquid marble swirls, rainbow colors melting together, ethereal glow, surreal abstract art",
    "Colorful ink drops in water, macro photography style, vivid magenta cyan yellow splashes, mesmerizing patterns",
    "Chaotic geometric shapes exploding, vaporwave aesthetic, holographic iridescent colors, glitch art vibes",
    "Bioluminescent jellyfish explosion of color, deep sea aurora, fantasy digital painting, otherworldly",
    "Graffiti paint bomb detonation, urban art chaos, spray paint texture, vibrant street art explosion",
    "Crystal shattering into rainbow fragments, prismatic light refraction, magical sparkles, fantasy art",
    "Aurora borealis meets fireworks, cosmic celebration, ethereal light show, breathtaking digital art",
    "Melting crayons abstract art, wax dripping in patterns, childhood nostalgia, macro detail, vivid colors",
    "Nebula birth explosion, cosmic dust clouds, stellar nursery, space art, vibrant purples pinks blues"
];

async function generateArt(customPrompt, outputName) {
    // Check for API token
    const token = process.env.REPLICATE_API_TOKEN;
    if (!token) {
        console.error('❌ Error: REPLICATE_API_TOKEN not found in .env file');
        process.exit(1);
    }

    const replicate = new Replicate({
        auth: token,
    });

    // Select or use custom prompt
    const prompt = customPrompt || CREATIVE_PROMPTS[Math.floor(Math.random() * CREATIVE_PROMPTS.length)];
    
    console.log('🎨 Generating creative art...');
    console.log(`📝 Prompt: ${prompt.substring(0, 80)}...`);

    try {
        const output = await replicate.run(
            "ideogram-ai/ideogram-v3-turbo",
            {
                input: {
                    prompt: prompt,
                    aspect_ratio: "1:1",
                    safety_tolerance: 2,
                    negative_prompt: "text, words, letters, watermark, signature, blurry, low quality"
                }
            }
        );

        // Output can be a URL string or array
        const imageUrl = Array.isArray(output) ? output[0] : output;
        
        if (!imageUrl) {
            console.error('❌ No image URL returned from API');
            process.exit(1);
        }

        console.log('✅ Image generated! Downloading...');

        // Create output directory
        const outputDir = path.join(projectRoot, 'public', 'images', 'generated');
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        // Download the image
        const response = await fetch(imageUrl);
        const arrayBuffer = await response.arrayBuffer();
        const buffer = Buffer.from(arrayBuffer);

        // Generate filename
        const timestamp = Date.now();
        const filename = outputName 
            ? `${outputName}.png`
            : `art-${timestamp}.png`;
        const outputPath = path.join(outputDir, filename);

        fs.writeFileSync(outputPath, buffer);

        console.log('');
        console.log('🖼️  Image saved!');
        console.log(`📁 Path: ${outputPath}`);
        console.log(`🔗 Web path: /images/generated/${filename}`);
        console.log('');
        console.log('Use in markdown:');
        console.log(`![creative art](/images/generated/${filename})`);

    } catch (error) {
        console.error('❌ Error generating image:', error.message);
        if (error.message.includes('401') || error.message.includes('Unauthorized')) {
            console.error('   Your API token may be invalid or expired.');
        }
        process.exit(1);
    }
}

// Parse command line arguments
const args = process.argv.slice(2);
const customPrompt = args[0] || null;
const outputName = args[1] || null;

generateArt(customPrompt, outputName);
