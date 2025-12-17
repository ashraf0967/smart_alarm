import numpy as np
import flet as ft
from typing import Dict, List
import json
import wave
import os
import time

class AlarmSoundManager:
    def __init__(self):
        self.volume = 0.5
        self.current_sound = "urgent"
        self.generated_files = {} # Stores paths to generated wav files
        self.sounds_dir = "assets/sounds"
        
        # Ensure directory exists
        if not os.path.exists(self.sounds_dir):
            os.makedirs(self.sounds_dir)
            
        # أسماء وعناوين الأصوات
        self.sound_names = {
            "gentle": "Gentle Morning",
            "urgent": "Urgent Alarm",
            "melodic": "Melodic Tone",
            "beeping": "Beeping Sound",
            "chirping": "Bird Chirping",
            "digital": "Digital Beep",
        }
        
        # Generators
        self.generators = {
            "gentle": self.create_gentle_alarm,
            "urgent": self.create_urgent_alarm,
            "melodic": self.create_melodic_alarm,
            "beeping": self.create_beeping_alarm,
            "chirping": self.create_chirping_alarm,
            "digital": self.create_digital_alarm,
        }
        
        # Generate files on startup
        self.generate_sound_files()
        
        # تحميل التفضيلات
        self.load_preferences()
    
    def generate_sound_files(self):
        """Generate .wav files for all sounds if they don't exist"""
        for sound_id, generator in self.generators.items():
            filename = f"{self.sounds_dir}/{sound_id}.wav"
            self.generated_files[sound_id] = filename
            
            if not os.path.exists(filename):
                print(f"Generating sound: {sound_id}...")
                audio_data = generator()
                self._save_wav(filename, audio_data)
                
    def _save_wav(self, filename, audio_data):
        """Save raw bytes to wav file"""
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2) # 2 bytes for int16
                wav_file.setframerate(44100)
                wav_file.writeframes(audio_data)
        except Exception as e:
            print(f"Error saving {filename}: {e}")

    # --- Sound Generators (Returning raw bytes of int16 PCM) ---
    # Kept from original code but cleaned up
    
    def create_gentle_alarm(self) -> bytes:
        sample_rate = 44100
        duration = 2.0 # Increased duration for loop smoothness
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        tone += 0.3 * np.sin(2 * np.pi * 554.37 * t)
        tone += 0.2 * np.sin(2 * np.pi * 659.25 * t)
        
        envelope = np.exp(-1 * t) # Slower decay
        audio = tone * envelope
        audio_normalized = audio * 32767 / np.max(np.abs(audio))
        return audio_normalized.astype(np.int16).tobytes()
    
    def create_urgent_alarm(self) -> bytes:
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        tone1 = np.sin(2 * np.pi * 880 * t)
        tone2 = np.sin(2 * np.pi * 440 * t)
        # Create a more annoying pattern
        tone = 0.5 * np.sign(np.sin(2 * np.pi * 5 * t)) * (tone1 + tone2) # Chopped
        
        audio_normalized = tone * 32767 / np.max(np.abs(tone) + 1e-9) # Avoid div by zero
        return audio_normalized.astype(np.int16).tobytes()
    
    def create_melodic_alarm(self) -> bytes:
        sample_rate = 44100
        audio = np.array([], dtype=np.int16)
        
        notes = [(440, 0.3), (523.25, 0.3), (659.25, 0.3), (783.99, 0.4)]
        
        for freq, duration in notes:
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-4 * t)
            note_audio = tone * envelope
            audio = np.concatenate([audio, note_audio])
        
        audio_normalized = audio * 32767 / np.max(np.abs(audio))
        return audio_normalized.astype(np.int16).tobytes()
    
    def create_beeping_alarm(self) -> bytes:
        sample_rate = 44100
        beep_duration = 0.2
        silence_duration = 0.2
        audio = np.array([], dtype=np.float32)
        
        for _ in range(3):
            t = np.linspace(0, beep_duration, int(sample_rate * beep_duration), False)
            beep = np.sin(2 * np.pi * 660 * t)
            audio = np.concatenate([audio, beep])
            
            silence = np.zeros(int(sample_rate * silence_duration))
            audio = np.concatenate([audio, silence])
        
        audio_normalized = audio * 32767 / np.max(np.abs(audio))
        return audio_normalized.astype(np.int16).tobytes()
    
    def create_chirping_alarm(self) -> bytes:
        sample_rate = 44100
        audio = np.array([], dtype=np.float32)
        
        for i in range(3):
            duration = 0.15 + i * 0.05
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            freq = 800 + 200 * np.sin(2 * np.pi * 8 * t)
            chirp = np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-5 * t)
            chirp_audio = chirp * envelope
            
            audio = np.concatenate([audio, chirp_audio])
            silence = np.zeros(int(sample_rate * 0.1))
            audio = np.concatenate([audio, silence])
        
        audio_normalized = audio * 32767 / np.max(np.abs(audio))
        return audio_normalized.astype(np.int16).tobytes()
    
    def create_digital_alarm(self) -> bytes:
        sample_rate = 44100
        duration = 0.2
        audio = np.array([], dtype=np.float32)
        
        for i in range(4):
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Square wave
            square_wave = np.where(np.sin(2 * np.pi * (800 + i * 100) * t) > 0, 1, -1)
            envelope = np.exp(-6 * t)
            digital_audio = square_wave * envelope
            
            audio = np.concatenate([audio, digital_audio])
            if i < 3:
                silence = np.zeros(int(sample_rate * 0.05))
                audio = np.concatenate([audio, silence])
        
        audio_normalized = audio * 32767 / np.max(np.abs(audio))
        return audio_normalized.astype(np.int16).tobytes()

    # --- Management and Preferences ---

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))
        self.save_preferences()
    
    def set_sound(self, sound_type: str):
        if sound_type in self.generated_files:
            self.current_sound = sound_type
            self.save_preferences()
            
    def get_current_sound_path(self):
        return self.generated_files.get(self.current_sound, self.generated_files["urgent"])

    def get_available_sounds(self) -> List[dict]:
        sounds_list = []
        for key, name in self.sound_names.items():
            sounds_list.append({
                "id": key,
                "name": name,
                "icon": self.get_sound_icon(key)
            })
        return sounds_list
    
    def get_sound_icon(self, sound_type: str) -> str:
        icons = {
            "gentle": ft.icons.MUSIC_NOTE,
            "urgent": ft.icons.NOTIFICATION_IMPORTANT,
            "melodic": ft.icons.PIANO,
            "beeping": ft.icons.TIMER,
            "chirping": ft.icons.NATURE,
            "digital": ft.icons.MEMORY,
        }
        return icons.get(sound_type, ft.icons.MUSIC_NOTE)
    
    def save_preferences(self, filename: str = "alarm_preferences.json"):
        preferences = {
            "current_sound": self.current_sound,
            "volume": self.volume,
            "last_used": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open(filename, 'w') as f:
                json.dump(preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def load_preferences(self, filename: str = "alarm_preferences.json"):
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    preferences = json.load(f)
                    self.current_sound = preferences.get("current_sound", "urgent")
                    self.volume = preferences.get("volume", 0.5)
        except Exception as e:
            print(f"Error loading preferences: {e}")