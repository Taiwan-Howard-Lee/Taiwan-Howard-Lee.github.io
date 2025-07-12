class GameCanvas {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.setupCanvas();
        this.setupResizeHandler();
    }

    setupCanvas() {
        this.resizeCanvas();
        this.canvas.style.cursor = 'crosshair';
    }

    setupResizeHandler() {
        window.addEventListener('resize', () => {
            this.resizeCanvas();
        });
    }

    resizeCanvas() {
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
    }

    drawBackground() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.width, this.height);

        // Draw four colored corners with smooth gradients
        this.drawCornerWithGradient(0, 0, this.width/2, this.height/2, '#0066CC', 'Blue'); // Top Left - Blue
        this.drawCornerWithGradient(this.width/2, 0, this.width/2, this.height/2, '#FFCC00', 'Yellow'); // Top Right - Yellow
        this.drawCornerWithGradient(0, this.height/2, this.width/2, this.height/2, '#00CC66', 'Green'); // Bottom Left - Green
        this.drawCornerWithGradient(this.width/2, this.height/2, this.width/2, this.height/2, '#CC0000', 'Red'); // Bottom Right - Red

        // Draw central neutral area
        this.drawCentralArea();

        // Draw center lines for visual separation
        this.drawCenterLines();

        // Draw zone labels
        this.drawZoneLabels();
    }

    drawCornerWithGradient(x, y, width, height, color) {
        // Create gradient from center to corner
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const cornerX = x + (x === 0 ? 0 : width);
        const cornerY = y + (y === 0 ? 0 : height);

        const gradient = this.ctx.createRadialGradient(
            centerX, centerY, 0,
            cornerX, cornerY, Math.sqrt(width * width + height * height)
        );

        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
        gradient.addColorStop(0.7, color + '80'); // Semi-transparent
        gradient.addColorStop(1, color);

        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(x, y, width, height);
    }

    drawCentralArea() {
        // Define central neutral area (circle in the middle) - 2x bigger
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const centralRadius = Math.min(this.width, this.height) * 0.3; // 30% of smaller dimension (2x bigger)

        // Draw central circle with neutral color
        const centralGradient = this.ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, centralRadius
        );
        centralGradient.addColorStop(0, '#FFFFFF');
        centralGradient.addColorStop(0.7, '#F5F5F5');
        centralGradient.addColorStop(1, '#E0E0E0');

        this.ctx.fillStyle = centralGradient;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, centralRadius, 0, Math.PI * 2);
        this.ctx.fill();

        // Draw border around central area
        this.ctx.strokeStyle = '#999999';
        this.ctx.lineWidth = 3;
        this.ctx.stroke();

        // Add "START" label in center
        this.ctx.fillStyle = '#666666';
        this.ctx.font = 'bold 16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('START', centerX, centerY - 5);
        this.ctx.font = 'bold 12px Arial';
        this.ctx.fillText('NEUTRAL ZONE', centerX, centerY + 10);
    }

    drawCenterLines() {
        this.ctx.strokeStyle = '#333333';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);

        // Vertical line
        this.ctx.beginPath();
        this.ctx.moveTo(this.width / 2, 0);
        this.ctx.lineTo(this.width / 2, this.height);
        this.ctx.stroke();

        // Horizontal line
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.height / 2);
        this.ctx.lineTo(this.width, this.height / 2);
        this.ctx.stroke();

        this.ctx.setLineDash([]); // Reset line dash
    }

    drawZoneLabels() {
        this.ctx.fillStyle = '#333333';
        this.ctx.font = 'bold 24px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = 3;

        // Blue zone (top-left)
        this.ctx.strokeText('ANALYSER', this.width / 4, 50);
        this.ctx.fillText('ANALYSER', this.width / 4, 50);

        // Yellow zone (top-right)
        this.ctx.strokeText('PLAYER', (this.width * 3) / 4, 50);
        this.ctx.fillText('PLAYER', (this.width * 3) / 4, 50);

        // Green zone (bottom-left)
        this.ctx.strokeText('KEEPER', this.width / 4, this.height - 30);
        this.ctx.fillText('KEEPER', this.width / 4, this.height - 30);

        // Red zone (bottom-right)
        this.ctx.strokeText('CARER', (this.width * 3) / 4, this.height - 30);
        this.ctx.fillText('CARER', (this.width * 3) / 4, this.height - 30);
    }

    // Helper method to determine which zone a point is in
    getZone(x, y) {
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const centralRadius = Math.min(this.width, this.height) * 0.3; // Match the larger central area

        // Check if point is in central neutral area
        const distanceFromCenter = Math.sqrt(
            Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2)
        );

        if (distanceFromCenter <= centralRadius) {
            return 'Neutral Zone';
        }

        // Check which colored zone the point is in
        if (x < centerX && y < centerY) return 'Blue - Analyser';
        if (x >= centerX && y < centerY) return 'Yellow - Player';
        if (x < centerX && y >= centerY) return 'Green - Keeper';
        if (x >= centerX && y >= centerY) return 'Red - Carer';

        return 'Center';
    }
}
