import os
import subprocess
from typing import List


class FluidSynth(object):
    
    def __init__(
        self, 
        sampling_rate: int = 44100,
        soundfont_path: str = "~/.fluidsynth/default_soundfont.sf2",
    ) -> None:
        self.sampling_rate = sampling_rate
        self.soundfont_path = os.path.expanduser(soundfont_path)
        
        if not os.path.exists(self.soundfont_path):
            self.download_sound_font()
        
    @staticmethod
    def execute_shell(cmd: List[str], surpress_error: bool = True) -> None:
        if surpress_error:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd)
    
    def download_sound_font(self) -> None:
        default_soundfont_url = "https://musical-artifacts.com/artifacts/538/Roland_SC-88.sf2"
        FluidSynth.execute_shell(f"wget {default_soundfont_url} -O {self.soundfont_path}")
    
    def midi_to_audio(self, midi_path: str, wav_path: str, quiet: bool = True) -> None:
        cmd = ["fluidsynth"]
        if quiet:
            cmd.append("--quiet")
        cmd = cmd + [
            "-ni", 
            "-F", wav_path, 
            "-r", str(self.sampling_rate), 
            self.soundfont_path, 
            midi_path
        ]
        stdout = FluidSynth.execute_shell(cmd)