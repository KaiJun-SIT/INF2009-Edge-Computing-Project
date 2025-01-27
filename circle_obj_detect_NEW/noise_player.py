import vlc
import os
import time

AUDIO_DIR = "noise/"

class Noise:
    def __init__(self, warning1, warning2, warning3):
        self.noise = vlc.MediaPlayer()
        self.warning1 = warning1
        self.warning2 = warning2
        self.warning3 = warning3
        self.current_audio = None
        self.is_audio_playing = False
        

    def setnoise(self, depth):
        self.depth = depth
        if self.depth is None:
            return
        # If audio is currently playing, don't interrupt it
        if self.noise.is_playing():
            return
            
        # Only set new audio if nothing is playing
        elif self.depth <= self.warning1: 
            if self.current_audio != "warning1":
                self.noise.stop()
                audio_path = os.path.join(AUDIO_DIR, "warning1.mp3")
                self.noise.set_media(vlc.Media(audio_path))
                self.noise.play()
                self.current_audio = "warning1"
            
        elif self.depth <= self.warning2:
            if self.current_audio != "warning2":
                self.noise.stop()
                audio_path = os.path.join(AUDIO_DIR, "warning2.mp3")
                self.noise.set_media(vlc.Media(audio_path))
                self.noise.play()
                self.current_audio = "warning2"
                
        elif self.depth <= self.warning3:
            if self.current_audio != "warning3":
                self.noise.stop()
                audio_path = os.path.join(AUDIO_DIR, "warning3.mp3")
                self.noise.set_media(vlc.Media(audio_path))
                self.noise.play()
                self.current_audio = "warning3"
                
        else:
            self.noise.stop()
            self.current_audio = None
            
    def play(self):
        self.noise.play()
        
    def stop(self):
        self.noise.stop()
        self.current_audio = None