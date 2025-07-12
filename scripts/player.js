class Player {
    constructor(username = 'Demo User', x = 400, y = 300) {
        this.username = username;
        this.x = x;
        this.y = y;
        this.size = 50; // bigger square size
        this.speed = 200; // pixels per second
        this.baseColor = '#FFFFFF';
        this.strokeColor = '#333333';
        this.keys = {};
        this.currentZone = 'Neutral Zone';

        // Square color collection properties
        this.collectedColors = []; // Can hold up to 4 colors
        this.maxColors = 4;

        // Animation properties
        this.bobOffset = 0;
        this.bobSpeed = 3;
        this.isMoving = false;
        this.collectAnimation = 0;
        this.isCollecting = false;
    }

    update(deltaTime, canvasWidth, canvasHeight) {
        let moved = false;

        // Handle movement based on pressed keys
        if (this.keys['w'] || this.keys['W'] || this.keys['ArrowUp']) {
            this.y -= this.speed * deltaTime;
            moved = true;
        }
        if (this.keys['s'] || this.keys['S'] || this.keys['ArrowDown']) {
            this.y += this.speed * deltaTime;
            moved = true;
        }
        if (this.keys['a'] || this.keys['A'] || this.keys['ArrowLeft']) {
            this.x -= this.speed * deltaTime;
            moved = true;
        }
        if (this.keys['d'] || this.keys['D'] || this.keys['ArrowRight']) {
            this.x += this.speed * deltaTime;
            moved = true;
        }

        // Boundary checking (using half size instead of radius)
        const halfSize = this.size / 2;
        this.x = Math.max(halfSize, Math.min(canvasWidth - halfSize, this.x));
        this.y = Math.max(halfSize, Math.min(canvasHeight - halfSize, this.y));

        // Update animations
        this.isMoving = moved;
        if (this.isMoving) {
            this.bobOffset += this.bobSpeed * deltaTime;
        }

        // Update collection animation
        if (this.isCollecting) {
            this.collectAnimation += deltaTime * 8; // Fast animation
            if (this.collectAnimation >= Math.PI * 2) {
                this.isCollecting = false;
                this.collectAnimation = 0;
            }
        }

        return moved;
    }

    draw(ctx) {
        // Calculate bob animation
        const bobY = this.isMoving ? Math.sin(this.bobOffset) * 2 : 0;
        const drawY = this.y + bobY;

        // Collection animation scale
        const collectScale = this.isCollecting ? 1 + Math.sin(this.collectAnimation) * 0.2 : 1;
        const squareSize = this.size * collectScale;

        // Draw shadow
        ctx.beginPath();
        ctx.ellipse(this.x, this.y + this.size/2 + 8, squareSize * 0.6, squareSize * 0.2, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fill();

        // Draw simple square
        this.drawSquare(ctx, this.x, drawY, squareSize);

        // Draw username with background
        this.drawUsername(ctx, this.x, drawY - squareSize/2 - 20);

        // Draw color collection indicator (positioned to avoid overlap)
        this.drawColorIndicator(ctx, this.x + squareSize/2 + 25, drawY - 10);
    }

    drawSquare(ctx, x, y, size) {
        const halfSize = size / 2;

        // Define square position
        const square = {
            x: x - halfSize,
            y: y - halfSize,
            width: size,
            height: size
        };

        // Draw square with collected colors
        this.drawSquareWithColors(ctx, square);

        // Draw smiley face
        this.drawSmileyFace(ctx, x, y, size);
    }

    drawSquareWithColors(ctx, square) {
        // Draw square with collected colors (up to 4 colors vertically)
        const numColors = this.collectedColors.length;
        const sectionHeight = square.height / 4; // Divide into 4 vertical sections

        // Draw each section
        for (let i = 0; i < 4; i++) {
            const sectionY = square.y + (i * sectionHeight);

            if (i < numColors) {
                // Draw collected color
                ctx.fillStyle = this.collectedColors[i];
            } else {
                // Draw default white for empty sections
                ctx.fillStyle = this.baseColor;
            }

            ctx.fillRect(square.x, sectionY, square.width, sectionHeight);
        }

        // Draw square border
        ctx.strokeStyle = this.strokeColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(square.x, square.y, square.width, square.height);

        // Draw dividing lines between sections
        ctx.strokeStyle = this.strokeColor;
        ctx.lineWidth = 2;
        for (let i = 1; i < 4; i++) {
            const lineY = square.y + (i * sectionHeight);
            ctx.beginPath();
            ctx.moveTo(square.x, lineY);
            ctx.lineTo(square.x + square.width, lineY);
            ctx.stroke();
        }
    }

    drawSmileyFace(ctx, x, y, size) {
        // Draw eyes
        ctx.fillStyle = '#333333';
        const eyeSize = size * 0.08;
        const eyeOffset = size * 0.15;

        // Left eye
        ctx.beginPath();
        ctx.arc(x - eyeOffset, y - eyeOffset/2, eyeSize, 0, Math.PI * 2);
        ctx.fill();

        // Right eye
        ctx.beginPath();
        ctx.arc(x + eyeOffset, y - eyeOffset/2, eyeSize, 0, Math.PI * 2);
        ctx.fill();

        // Smile
        ctx.strokeStyle = '#333333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y + eyeOffset/3, size * 0.2, 0, Math.PI);
        ctx.stroke();
    }

    drawUsername(ctx, x, y) {
        // Measure text for background
        ctx.font = 'bold 14px Arial';
        const textWidth = ctx.measureText(this.username).width;
        const padding = 6;

        // Draw background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(
            x - textWidth/2 - padding,
            y - 8,
            textWidth + padding * 2,
            16
        );

        // Draw text
        ctx.fillStyle = '#FFFFFF';
        ctx.textAlign = 'center';
        ctx.fillText(this.username, x, y + 4);
    }

    drawColorIndicator(ctx, x, y) {
        // Draw compact collected colors indicator
        const indicatorWidth = 50;
        const indicatorHeight = 45;

        ctx.font = 'bold 10px Arial';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(x - 5, y - 20, indicatorWidth, indicatorHeight);

        ctx.fillStyle = '#FFFFFF';
        ctx.textAlign = 'left';
        ctx.fillText('Traits:', x, y - 10);
        ctx.fillText(`${this.collectedColors.length}/4`, x, y + 2);

        // Draw small color squares in a compact 2x2 grid
        for (let i = 0; i < Math.min(this.collectedColors.length, 4); i++) {
            const row = Math.floor(i / 2);
            const col = i % 2;
            const squareX = x + 5 + (col * 14);
            const squareY = y + 8 + (row * 12);

            ctx.fillStyle = this.collectedColors[i];
            ctx.fillRect(squareX, squareY, 10, 10);
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.strokeRect(squareX, squareY, 10, 10);
        }
    }

    // Method to handle key press
    onKeyDown(key) {
        this.keys[key] = true;

        // Handle Enter key for color collection
        if (key === 'Enter') {
            this.collectColor();
        }
    }

    // Method to handle key release
    onKeyUp(key) {
        this.keys[key] = false;
    }

    // Color collection method
    collectColor() {
        const zoneColor = this.getZoneColor(this.currentZone);

        if (zoneColor) {
            if (this.collectedColors.length < this.maxColors) {
                // Add new color
                this.collectedColors.push(zoneColor);
                this.isCollecting = true;

                // Play collect sound
                if (window.audioManager) {
                    window.audioManager.playCollectSound();
                }

                // Create particle burst effect
                if (window.particleSystem) {
                    window.particleSystem.createCollectionBurst(this.x, this.y, zoneColor);
                }

                console.log(`✨ Collected ${this.currentZone} trait! (${this.collectedColors.length}/${this.maxColors})`);
            } else {
                // Clear all colors when at max capacity
                this.collectedColors = [];
                this.isCollecting = true;

                // Play clear sound
                if (window.audioManager) {
                    window.audioManager.playClearSound();
                }

                console.log('🔄 Square full! Cleared all personality traits');
            }
        } else {
            // If in neutral zone or center, clear colors
            this.collectedColors = [];
            this.isCollecting = true;

            // Play clear sound
            if (window.audioManager) {
                window.audioManager.playClearSound();
            }

            if (this.currentZone === 'Neutral Zone') {
                console.log('🔄 Reset in neutral zone');
            } else {
                console.log('🔄 Reset in center area');
            }
        }
    }

    // Get color based on zone
    getZoneColor(zone) {
        if (zone.includes('Blue')) return '#0066CC';
        if (zone.includes('Yellow')) return '#FFCC00';
        if (zone.includes('Green')) return '#00CC66';
        if (zone.includes('Red')) return '#CC0000';
        return null; // Neutral zone or center - no color
    }

    // Get current position
    getPosition() {
        return { x: this.x, y: this.y };
    }

    // Set zone information
    setCurrentZone(zone) {
        this.currentZone = zone;
    }

    // Get collected colors info
    getCollectedColors() {
        return {
            colors: [...this.collectedColors],
            count: this.collectedColors.length,
            maxColors: this.maxColors
        };
    }
}
