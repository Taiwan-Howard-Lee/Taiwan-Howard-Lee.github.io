@echo off
REM 🚀 GitHub Pages Deployment Script for Windows
REM Repository: Taiwan-Howard-Lee/MVP_click_colour

echo 🎮 Deploying Personality Arena to GitHub Pages...
echo 📁 Repository: Taiwan-Howard-Lee/MVP_click_colour
echo.

REM Check if we're in the right directory
if not exist "index.html" (
    echo ❌ Error: index.html not found. Please run this script from the project root directory.
    pause
    exit /b 1
)

echo ✅ Found index.html - we're in the right directory

REM Check if git is initialized
if not exist ".git" (
    echo 🔧 Initializing git repository...
    git init
    echo ✅ Git repository initialized
) else (
    echo ✅ Git repository already exists
)

REM Add all files
echo 📦 Adding all files to git...
git add .

REM Commit changes
echo 💾 Committing changes...
git commit -m "Deploy Personality Arena v%date:~-4,4%%date:~-10,2%%date:~-7,2%-%time:~0,2%%time:~3,2%%time:~6,2%"

REM Check if remote exists
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo 🔗 Adding GitHub remote...
    git remote add origin https://github.com/Taiwan-Howard-Lee/MVP_click_colour.git
    echo ✅ Remote origin added
) else (
    echo ✅ Remote origin already configured
)

REM Push to GitHub
echo 🚀 Pushing to GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo 🎉 Deployment successful!
    echo.
    echo 📍 Your game will be available at:
    echo    https://taiwan-howard-lee.github.io/MVP_click_colour/
    echo.
    echo ⏱️  It may take 2-10 minutes for GitHub Pages to update
    echo.
    echo 🔧 To enable GitHub Pages (if not already done):
    echo    1. Go to: https://github.com/Taiwan-Howard-Lee/MVP_click_colour
    echo    2. Click 'Settings' tab
    echo    3. Scroll to 'Pages' section
    echo    4. Set Source to 'Deploy from a branch'
    echo    5. Select 'main' branch and '/ (root)' folder
    echo    6. Click 'Save'
    echo.
    echo 🎮 Ready for your boss demo!
) else (
    echo.
    echo ❌ Deployment failed. Please check:
    echo    - Your GitHub credentials are set up
    echo    - The repository exists: Taiwan-Howard-Lee/MVP_click_colour
    echo    - You have push permissions to the repository
    echo.
    echo 💡 You can also deploy manually:
    echo    1. Create the repository on GitHub
    echo    2. Upload all files via GitHub web interface
    echo    3. Enable GitHub Pages in repository settings
)

echo.
echo Press any key to continue...
pause >nul
