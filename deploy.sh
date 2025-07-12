#!/bin/bash

# 🚀 GitHub Pages Deployment Script
# Repository: Taiwan-Howard-Lee/MVP_click_colour

echo "🎮 Deploying Personality Arena to GitHub Pages..."
echo "📁 Repository: Taiwan-Howard-Lee/MVP_click_colour"
echo ""

# Check if we're in the right directory
if [ ! -f "index.html" ]; then
    echo "❌ Error: index.html not found. Please run this script from the project root directory."
    exit 1
fi

echo "✅ Found index.html - we're in the right directory"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "🔧 Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files
echo "📦 Adding all files to git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "💾 Committing changes..."
    git commit -m "Deploy Personality Arena v$(date +%Y%m%d-%H%M%S)"
    echo "✅ Changes committed"
fi

# Check if remote exists
if git remote get-url origin > /dev/null 2>&1; then
    echo "✅ Remote origin already configured"
else
    echo "🔗 Adding GitHub remote..."
    git remote add origin https://github.com/Taiwan-Howard-Lee/MVP_click_colour.git
    echo "✅ Remote origin added"
fi

# Push to GitHub
echo "🚀 Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Deployment successful!"
    echo ""
    echo "📍 Your game will be available at:"
    echo "   https://taiwan-howard-lee.github.io/MVP_click_colour/"
    echo ""
    echo "⏱️  It may take 2-10 minutes for GitHub Pages to update"
    echo ""
    echo "🔧 To enable GitHub Pages (if not already done):"
    echo "   1. Go to: https://github.com/Taiwan-Howard-Lee/MVP_click_colour"
    echo "   2. Click 'Settings' tab"
    echo "   3. Scroll to 'Pages' section"
    echo "   4. Set Source to 'Deploy from a branch'"
    echo "   5. Select 'main' branch and '/ (root)' folder"
    echo "   6. Click 'Save'"
    echo ""
    echo "🎮 Ready for your boss demo!"
else
    echo ""
    echo "❌ Deployment failed. Please check:"
    echo "   - Your GitHub credentials are set up"
    echo "   - The repository exists: Taiwan-Howard-Lee/MVP_click_colour"
    echo "   - You have push permissions to the repository"
    echo ""
    echo "💡 You can also deploy manually:"
    echo "   1. Create the repository on GitHub"
    echo "   2. Upload all files via GitHub web interface"
    echo "   3. Enable GitHub Pages in repository settings"
fi
