class ParticleSystem {
    constructor() {
        this.particles = [];
        this.container = document.getElementById('particles-container');
        this.maxParticles = 50;
        this.spawnRate = 0.3; // particles per frame
        this.lastSpawn = 0;
        
        this.init();
    }
    
    init() {
        // Start particle generation
        this.generateParticles();
        
        // Create initial burst of particles
        for (let i = 0; i < 20; i++) {
            setTimeout(() => {
                this.createParticle();
            }, i * 100);
        }
    }
    
    generateParticles() {
        setInterval(() => {
            if (this.particles.length < this.maxParticles && Math.random() < this.spawnRate) {
                this.createParticle();
            }
            this.cleanupParticles();
        }, 100);
    }
    
    createParticle(zone = null) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random position
        const x = Math.random() * window.innerWidth;
        const y = window.innerHeight + 10;
        
        // Zone-specific styling
        if (zone) {
            particle.classList.add(zone.toLowerCase());
        } else {
            // Random zone for ambient particles
            const zones = ['analyser', 'player', 'keeper', 'carer'];
            const randomZone = zones[Math.floor(Math.random() * zones.length)];
            particle.classList.add(randomZone);
        }
        
        // Random size and animation duration
        const size = 2 + Math.random() * 4;
        const duration = 6 + Math.random() * 4;
        const delay = Math.random() * 2;
        
        particle.style.left = x + 'px';
        particle.style.top = y + 'px';
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        particle.style.animationDuration = duration + 's';
        particle.style.animationDelay = delay + 's';
        
        // Add random horizontal drift
        const drift = (Math.random() - 0.5) * 100;
        particle.style.setProperty('--drift', drift + 'px');
        
        this.container.appendChild(particle);
        this.particles.push({
            element: particle,
            createdAt: Date.now(),
            duration: (duration + delay) * 1000
        });
        
        // Remove particle after animation
        setTimeout(() => {
            this.removeParticle(particle);
        }, (duration + delay) * 1000);
    }
    
    createCollectionBurst(x, y, color) {
        // Create burst effect when collecting colors
        for (let i = 0; i < 8; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle burst-particle';
            
            const angle = (i / 8) * Math.PI * 2;
            const velocity = 50 + Math.random() * 30;
            const size = 3 + Math.random() * 3;
            
            particle.style.left = x + 'px';
            particle.style.top = y + 'px';
            particle.style.width = size + 'px';
            particle.style.height = size + 'px';
            particle.style.background = color;
            particle.style.boxShadow = `0 0 10px ${color}`;
            
            // Calculate end position
            const endX = x + Math.cos(angle) * velocity;
            const endY = y + Math.sin(angle) * velocity;
            
            particle.style.setProperty('--end-x', endX + 'px');
            particle.style.setProperty('--end-y', endY + 'px');
            
            // Add burst animation
            particle.style.animation = 'burstParticle 0.8s ease-out forwards';
            
            this.container.appendChild(particle);
            
            setTimeout(() => {
                this.removeParticle(particle);
            }, 800);
        }
    }
    
    createZoneTransitionEffect(zone) {
        // Create special effect when entering a new zone
        const colors = {
            'Blue - Analyser': '#0066CC',
            'Yellow - Player': '#FFCC00',
            'Green - Keeper': '#00CC66',
            'Red - Carer': '#CC0000',
            'Neutral Zone': '#FFFFFF'
        };
        
        const color = colors[zone] || '#FFFFFF';
        
        // Create ripple effect
        for (let i = 0; i < 5; i++) {
            setTimeout(() => {
                this.createParticle(zone.split(' - ')[0].toLowerCase());
            }, i * 100);
        }
    }
    
    removeParticle(particle) {
        if (particle && particle.parentNode) {
            particle.parentNode.removeChild(particle);
        }
        
        this.particles = this.particles.filter(p => p.element !== particle);
    }
    
    cleanupParticles() {
        const now = Date.now();
        this.particles = this.particles.filter(particle => {
            if (now - particle.createdAt > particle.duration + 1000) {
                this.removeParticle(particle.element);
                return false;
            }
            return true;
        });
    }
    
    setIntensity(intensity) {
        // Adjust particle spawn rate based on intensity (0-1)
        this.spawnRate = 0.1 + (intensity * 0.5);
    }
    
    clear() {
        this.particles.forEach(particle => {
            this.removeParticle(particle.element);
        });
        this.particles = [];
    }
}

// Add CSS for burst particles
const style = document.createElement('style');
style.textContent = `
    .burst-particle {
        animation: burstParticle 0.8s ease-out forwards !important;
    }
    
    @keyframes burstParticle {
        0% {
            transform: translate(0, 0) scale(1);
            opacity: 1;
        }
        100% {
            transform: translate(var(--end-x, 0), var(--end-y, 0)) scale(0);
            opacity: 0;
        }
    }
    
    .particle {
        transform: translateX(var(--drift, 0));
    }
`;
document.head.appendChild(style);

// Global particle system instance
window.particleSystem = new ParticleSystem();
