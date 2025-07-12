class AudioManager {
    constructor() {
        this.sounds = {};
        this.enabled = true;
        this.volume = 0.3;
        this.initializeSounds();
    }
    
    initializeSounds() {
        // Create audio contexts for Web Audio API
        this.audioContext = null;
        
        // Initialize with fallback sounds using Web Audio API
        this.createSynthSounds();
        
        // Try to load actual audio files if they exist
        this.loadAudioFiles();
    }
    
    createSynthSounds() {
        // Create synthetic sounds using Web Audio API
        this.sounds = {
            move: () => this.playTone(200, 0.1, 'sine'),
            collect: () => this.playChord([440, 554, 659], 0.3, 'sine'),
            clear: () => this.playTone(150, 0.5, 'sawtooth'),
            zoneEnter: (frequency) => this.playTone(frequency, 0.2, 'triangle')
        };
    }
    
    loadAudioFiles() {
        // Try to load actual audio files
        const audioElements = {
            move: document.getElementById('moveSound'),
            collect: document.getElementById('collectSound'),
            clear: document.getElementById('clearSound'),
            ambient: document.getElementById('ambientSound')
        };
        
        // Override synthetic sounds with audio files if they load successfully
        Object.keys(audioElements).forEach(key => {
            const audio = audioElements[key];
            if (audio) {
                audio.volume = this.volume;
                audio.addEventListener('canplaythrough', () => {
                    if (key !== 'ambient') {
                        this.sounds[key] = () => {
                            if (this.enabled) {
                                audio.currentTime = 0;
                                audio.play().catch(() => {});
                            }
                        };
                    }
                });
            }
        });
    }
    
    getAudioContext() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        return this.audioContext;
    }
    
    playTone(frequency, duration, waveType = 'sine') {
        if (!this.enabled) return;
        
        try {
            const ctx = this.getAudioContext();
            const oscillator = ctx.createOscillator();
            const gainNode = ctx.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(ctx.destination);
            
            oscillator.frequency.setValueAtTime(frequency, ctx.currentTime);
            oscillator.type = waveType;
            
            gainNode.gain.setValueAtTime(0, ctx.currentTime);
            gainNode.gain.linearRampToValueAtTime(this.volume * 0.3, ctx.currentTime + 0.01);
            gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
            
            oscillator.start(ctx.currentTime);
            oscillator.stop(ctx.currentTime + duration);
        } catch (error) {
            console.log('Audio not available');
        }
    }
    
    playChord(frequencies, duration, waveType = 'sine') {
        if (!this.enabled) return;
        
        frequencies.forEach((freq, index) => {
            setTimeout(() => {
                this.playTone(freq, duration, waveType);
            }, index * 50);
        });
    }
    
    playMoveSound() {
        if (this.sounds.move) {
            this.sounds.move();
        }
    }
    
    playCollectSound() {
        if (this.sounds.collect) {
            this.sounds.collect();
        }
    }
    
    playClearSound() {
        if (this.sounds.clear) {
            this.sounds.clear();
        }
    }
    
    playZoneEnterSound(zone) {
        const frequencies = {
            'Blue - Analyser': 330,
            'Yellow - Player': 440,
            'Green - Keeper': 523,
            'Red - Carer': 659,
            'Neutral Zone': 220
        };
        
        const frequency = frequencies[zone] || 220;
        if (this.sounds.zoneEnter) {
            this.sounds.zoneEnter(frequency);
        }
    }
    
    startAmbient() {
        const ambient = document.getElementById('ambientSound');
        if (ambient && this.enabled) {
            ambient.volume = this.volume * 0.2;
            ambient.play().catch(() => {});
        }
    }
    
    stopAmbient() {
        const ambient = document.getElementById('ambientSound');
        if (ambient) {
            ambient.pause();
        }
    }
    
    toggle() {
        this.enabled = !this.enabled;
        if (!this.enabled) {
            this.stopAmbient();
        }
        return this.enabled;
    }
    
    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
    }
}

// Global audio manager instance
window.audioManager = new AudioManager();
