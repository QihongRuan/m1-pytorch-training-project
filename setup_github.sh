#!/bin/bash

echo "🚀 Setting up GitHub repository for M1 PyTorch Training Project"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Please run this script from the project directory"
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo "📊 Repository contents:"
ls -la

echo ""
echo "🔗 Attempting to create GitHub repository..."

# Try using curl with GitHub API (requires personal access token)
echo "📝 Creating repository using GitHub API..."
echo "ℹ️  You'll need to provide your GitHub credentials when prompted"

# Create the repository
curl -u "QihongRuan" \
     -X POST \
     -H "Accept: application/vnd.github.v3+json" \
     https://api.github.com/user/repos \
     -d '{
       "name": "m1-pytorch-training-project",
       "description": "M1-optimized PyTorch training with real-world CIFAR-10 classification - Created with Claude Code",
       "private": false,
       "has_issues": true,
       "has_projects": true,
       "has_wiki": true
     }'

echo ""
echo "🚀 Now pushing files to GitHub..."

# Push to GitHub
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Repository created and files pushed!"
    echo "🌐 View your repository at: https://github.com/QihongRuan/m1-pytorch-training-project"
    echo ""
    echo "📊 Repository includes:"
    echo "   • M1-optimized PyTorch training scripts"
    echo "   • Real-time monitoring tools"  
    echo "   • Comprehensive documentation"
    echo "   • CIFAR-10 classification project"
else
    echo "❌ Push failed. Please check your GitHub authentication."
    echo "💡 You may need to set up a Personal Access Token"
    echo "🔗 Visit: https://github.com/settings/tokens"
fi