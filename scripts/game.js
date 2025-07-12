class Game {
    constructor() {
        this.canvas = new GameCanvas('gameCanvas');
        this.player = new Player('Demo User');
        this.lastTime = 0;
        this.currentZoneElement = document.getElementById('current-zone');
        this.colorSlotsElement = document.getElementById('color-slots');
        this.lastZone = '';

        this.setupInputHandlers();
        this.gameLoop();

        // Initialize audio and particles
        this.initializeAudio();
        this.updateUI();

        console.log('🎮 PERSONALITY ARENA LOADED!');
        console.log('🎯 Use WASD or Arrow Keys to move');
        console.log('⚡ Press ENTER to collect personality colors!');
        console.log('🎵 Audio and visual effects enabled');
    }

    initializeAudio() {
        // Start ambient sound after user interaction
        document.addEventListener('click', () => {
            if (window.audioManager) {
                window.audioManager.startAmbient();
            }
        }, { once: true });

        document.addEventListener('keydown', () => {
            if (window.audioManager) {
                window.audioManager.startAmbient();
            }
        }, { once: true });
    }

    setupInputHandlers() {
        // Keyboard event listeners
        document.addEventListener('keydown', (event) => {
            this.player.onKeyDown(event.key);
            event.preventDefault(); // Prevent page scrolling
        });

        document.addEventListener('keyup', (event) => {
            this.player.onKeyUp(event.key);
            event.preventDefault();
        });

        // Canvas click to move (optional feature)
        this.canvas.canvas.addEventListener('click', (event) => {
            const rect = this.canvas.canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            // Smooth movement towards clicked point (simple implementation)
            this.movePlayerTowards(x, y);
        });

        // Prevent context menu on right click
        this.canvas.canvas.addEventListener('contextmenu', (event) => {
            event.preventDefault();
        });

        // Touch controls for mobile
        this.setupTouchControls();

        // Focus canvas for keyboard input
        this.canvas.canvas.setAttribute('tabindex', '0');
        this.canvas.canvas.focus();
    }

    setupTouchControls() {
        let touchStartX = 0;
        let touchStartY = 0;
        let isTouching = false;

        this.canvas.canvas.addEventListener('touchstart', (event) => {
            event.preventDefault();
            const touch = event.touches[0];
            const rect = this.canvas.canvas.getBoundingClientRect();
            touchStartX = touch.clientX - rect.left;
            touchStartY = touch.clientY - rect.top;
            isTouching = true;
        });

        this.canvas.canvas.addEventListener('touchmove', (event) => {
            event.preventDefault();
            if (!isTouching) return;

            const touch = event.touches[0];
            const rect = this.canvas.canvas.getBoundingClientRect();
            const currentX = touch.clientX - rect.left;
            const currentY = touch.clientY - rect.top;

            // Calculate movement direction
            const deltaX = currentX - touchStartX;
            const deltaY = currentY - touchStartY;
            const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

            if (distance > 20) { // Minimum distance for movement
                // Simulate key presses based on touch direction
                const angle = Math.atan2(deltaY, deltaX);
                const degrees = angle * 180 / Math.PI;

                // Clear all movement keys first
                this.player.keys = {};

                // Set movement based on angle
                if (degrees >= -45 && degrees <= 45) {
                    this.player.keys['d'] = true; // Right
                } else if (degrees >= 45 && degrees <= 135) {
                    this.player.keys['s'] = true; // Down
                } else if (degrees >= 135 || degrees <= -135) {
                    this.player.keys['a'] = true; // Left
                } else {
                    this.player.keys['w'] = true; // Up
                }

                touchStartX = currentX;
                touchStartY = currentY;
            }
        });

        this.canvas.canvas.addEventListener('touchend', (event) => {
            event.preventDefault();
            isTouching = false;
            // Clear all movement keys
            this.player.keys = {};
        });

        // Double tap to collect colors
        let lastTap = 0;
        this.canvas.canvas.addEventListener('touchend', () => {
            const currentTime = new Date().getTime();
            const tapLength = currentTime - lastTap;
            if (tapLength < 500 && tapLength > 0) {
                // Double tap detected
                this.player.collectColor();
            }
            lastTap = currentTime;
        });
    }

    gameLoop(currentTime = 0) {
        const deltaTime = (currentTime - this.lastTime) / 1000;
        this.lastTime = currentTime;

        // Update game state
        this.update(deltaTime);

        // Render everything
        this.render();

        // Continue loop
        requestAnimationFrame((time) => this.gameLoop(time));
    }

    update(deltaTime) {
        // Update player
        const playerMoved = this.player.update(deltaTime, this.canvas.width, this.canvas.height);

        // Update zone information if player moved
        if (playerMoved) {
            this.updateCurrentZone();
            // Play movement sound occasionally
            if (Math.random() < 0.1 && window.audioManager) {
                window.audioManager.playMoveSound();
            }
        }

        // Update UI
        this.updateUI();
    }

    render() {
        // Clear and draw background
        this.canvas.drawBackground();

        // Draw player
        this.player.draw(this.canvas.ctx);
    }

    updateCurrentZone() {
        const position = this.player.getPosition();
        const zone = this.canvas.getZone(position.x, position.y);

        if (zone !== this.player.currentZone) {
            this.player.setCurrentZone(zone);

            // Play zone enter sound
            if (window.audioManager) {
                window.audioManager.playZoneEnterSound(zone);
            }

            // Create particle effect
            if (window.particleSystem) {
                window.particleSystem.createZoneTransitionEffect(zone);
            }

            this.lastZone = zone;
        }
    }

    updateUI() {
        // Update current zone display
        if (this.currentZoneElement) {
            const zone = this.player.currentZone;

            // Only update if zone changed
            if (zone !== this.lastZone) {
                this.currentZoneElement.textContent = zone;

                // Update zone color
                const colors = {
                    'Blue - Analyser': '#0066CC',
                    'Yellow - Player': '#FFCC00',
                    'Green - Keeper': '#00CC66',
                    'Red - Carer': '#CC0000',
                    'Neutral Zone': '#FFFFFF'
                };

                this.currentZoneElement.style.color = colors[zone] || '#FFFFFF';
                this.currentZoneElement.style.borderColor = colors[zone] || '#FFFFFF';

                // Add transition animation
                this.currentZoneElement.classList.add('zone-transition');
                setTimeout(() => {
                    this.currentZoneElement.classList.remove('zone-transition');
                }, 500);

                this.lastZone = zone;
            }
        }

        // Update color collection display
        this.updateColorSlots();
    }

    updateColorSlots() {
        if (!this.colorSlotsElement) return;

        const colorInfo = this.player.getCollectedColors();
        const slots = this.colorSlotsElement.children;

        // Update each slot
        for (let i = 0; i < slots.length; i++) {
            const slot = slots[i];

            if (i < colorInfo.colors.length) {
                // Slot has a color
                slot.className = 'color-slot collected';
                slot.style.backgroundColor = colorInfo.colors[i];
                slot.style.borderColor = colorInfo.colors[i];
                slot.style.color = colorInfo.colors[i]; // For currentColor in CSS
            } else {
                // Empty slot
                slot.className = 'color-slot empty';
                slot.style.backgroundColor = '';
                slot.style.borderColor = '';
                slot.style.color = '';
            }
        }
    }

    getZoneColor(zone) {
        if (zone.includes('Blue')) return '#0066CC';
        if (zone.includes('Yellow')) return '#FFCC00';
        if (zone.includes('Green')) return '#00CC66';
        if (zone.includes('Red')) return '#CC0000';
        return '#667eea';
    }

    movePlayerTowards(targetX, targetY) {
        // Simple click-to-move implementation
        const currentPos = this.player.getPosition();
        const dx = targetX - currentPos.x;
        const dy = targetY - currentPos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > 5) { // Only move if click is far enough
            const moveX = (dx / distance) * 3; // Move 3 pixels per frame
            const moveY = (dy / distance) * 3;

            // Animate movement over several frames
            const animateMove = () => {
                const pos = this.player.getPosition();
                const newDx = targetX - pos.x;
                const newDy = targetY - pos.y;
                const newDistance = Math.sqrt(newDx * newDx + newDy * newDy);

                if (newDistance > 5) {
                    this.player.x += moveX;
                    this.player.y += moveY;

                    // Keep within bounds
                    this.player.x = Math.max(this.player.radius,
                        Math.min(this.canvas.width - this.player.radius, this.player.x));
                    this.player.y = Math.max(this.player.radius,
                        Math.min(this.canvas.height - this.player.radius, this.player.y));

                    requestAnimationFrame(animateMove);
                }
            };

            animateMove();
        }
    }

    // Method to get current game state (useful for demos)
    getGameState() {
        return {
            playerPosition: this.player.getPosition(),
            currentZone: this.player.currentZone,
            canvasSize: { width: this.canvas.width, height: this.canvas.height }
        };
    }
}

// Version Control System
class VersionController {
    constructor() {
        this.isMobileMode = this.detectMobile();
        this.setupVersionToggle();
        this.applyInitialMode();
    }

    detectMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
               window.innerWidth <= 768;
    }

    setupVersionToggle() {
        const toggleBtn = document.getElementById('version-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                this.toggleMode();
            });
        }
    }

    toggleMode() {
        this.isMobileMode = !this.isMobileMode;
        this.applyMode();
    }

    applyInitialMode() {
        this.applyMode();
    }

    applyMode() {
        const container = document.querySelector('.arena-container');
        const toggleBtn = document.getElementById('version-toggle');
        const controlsHint = document.getElementById('controls-hint');

        if (this.isMobileMode) {
            container.classList.add('mobile-mode');
            if (toggleBtn) toggleBtn.textContent = '🖥️ Desktop Mode';
            if (controlsHint) {
                controlsHint.innerHTML = '<span class="key">SWIPE</span> Move • <span class="key">DOUBLE TAP</span> Collect • <span class="key">TAP</span> Navigate';
            }
            console.log('📱 Mobile mode activated');
        } else {
            container.classList.remove('mobile-mode');
            if (toggleBtn) toggleBtn.textContent = '📱 Mobile Mode';
            if (controlsHint) {
                controlsHint.innerHTML = '<span class="key">WASD</span> Move • <span class="key">ENTER</span> Collect • <span class="key">CLICK</span> Navigate';
            }
            console.log('🖥️ Desktop mode activated');
        }

        // Trigger canvas resize if game exists
        if (window.game && window.game.canvas) {
            window.game.canvas.resizeCanvas();
        }
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize version controller first
    const versionController = new VersionController();
    window.versionController = versionController;

    // Then initialize the game
    const game = new Game();

    // Make game accessible globally for debugging
    window.game = game;

    // Add some demo instructions
    console.log('=== Team Building Game Demo ===');
    console.log('Available commands:');
    console.log('- game.getGameState() - Get current game state');
    console.log('- versionController.toggleMode() - Switch between mobile/desktop');
    console.log('- Use WASD or Arrow Keys to move');
    console.log('- Click on canvas to move to that location');
});
