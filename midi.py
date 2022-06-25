import hashlib
import logging
import multiprocessing as mp
import os
from random import sample
import time
from typing import List, Optional, Tuple

import mido
import numpy as np
import pandas as pd
from fluidsynth import FluidSynth
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm.auto import tqdm

from midi_enum import MidiNote, MidiProgram
        

class MidiNotes(object):
    
    def __init__(self) -> None:
        self.notes = [note.value for note in list(MidiNote)]
        self.init_midi_notes()
        
    def get_midi_idx(self, note: MidiNote, position: Optional[int] = None) -> int:
        midi_notes = self.note_mapper[note.value]
        position = np.random.choice(len(midi_notes)) if position is None else position
        return midi_notes[position]
    
    def init_midi_notes(self) -> None:
        self.note_mapper = {note: [] for note in self.notes}
        
        # midi notes ranged from 21-128
        # ref. https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
        for i in range(21, 128):
            for note_idx, note in enumerate(self.notes):
                if i % len(self.notes) == note_idx:
                    self.note_mapper[note].append(i)
                    
                    
class MidiGenerator(object):
    def __init__(
        self, 
        max_duration: int = 1.5,  # in sec
        sampling_rate: int = 22050,  # top midi freq = 12543.85 Hz
        tmp_path: str = "tmp"
    ) -> None:
        self.tmp_path = tmp_path
        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.fluid_synth = FluidSynth(sampling_rate=self.sampling_rate)
        
        self.midi_notes = MidiNotes()
        self.programs = list(MidiProgram)
        
    def get_random_duration(
        self, 
        start_offset: Tuple[float] = (0.1, 0.3), 
        end_offset: Tuple[float] = (0.4, 0.9)
    ) -> Tuple[int, int]:
        """Randomly generate leading_silence_duration, and note_duration"""
        max_dur = int(self.max_duration * 1000 * np.random.uniform(*end_offset)) # convert sec -> ms
        leading_silence_duration = np.random.randint(low=50, high=max_dur * np.random.uniform(*start_offset))  # min leading silence of 100 ms
        note_duration = max_dur - leading_silence_duration
        return leading_silence_duration, note_duration
        
    def post_process_wav(
        self, 
        wav_path: str,
    ) -> np.ndarray:
        sr, wav = wavfile.read(wav_path)
        assert sr == self.sampling_rate, f"Loaded sampling rate mismatch. Expecting {self.sampling_rate}, but got {sr}"
        
        # preprocess wav here
        wav = wav.mean(-1)  # convert wav to mono
        wav = wav / wav.max()  # normalize audio
        wav = wav * np.random.uniform(low=0.75, high=1.)  # random audio level
        wav = wav + np.random.normal(
            scale=np.random.uniform(low=1e-5, high=1e-2), 
            size=wav.shape
        )  # inject some small noise
    
        return wav.astype(np.float32)  # for 32-bit floating point save
    
    def __clear_tmp(self, save_path: str) -> None:
        for extension in ["wav", "mid"]:
            if os.path.exists(save_path + extension):
                os.remove(save_path + extension)
                
    def _midi_to_wav(self, midi_path: str, wav_path: str) -> None:
        """Converts MIDI file to WAV"""
        self.fluid_synth.midi_to_audio(midi_path, wav_path)  # convert .mid -> .wav
        
    def save_wav(self, wav: np.ndarray, wav_path: str) -> None:
        wavfile.write(wav_path, self.sampling_rate, wav)

    @staticmethod
    def apply_sha256(name: str) -> str:
        """Apply sha256 hash to string. This will be used as a filename to prevent multiprocessing collision"""
        return str(int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16) % 1e8)

    @staticmethod
    def sec_to_hour(second: int, return_string: bool = True) -> Tuple[int, int , float]:
        minutes, seconds = divmod(second, 60)
        hours, minutes = divmod(minutes, 60)
        if return_string:
            return f"{int(hours)} hour(s) {int(minutes)} minutes {seconds:.4f} seconds"
        return int(hours), int(minutes), int(seconds)
                    
    def generate_midi_note(
        self, 
        note: MidiNote, 
        program: MidiProgram = 0,
        note_position: Optional[int] = None,
        leading_silence_duration: Optional[int] = None,  # in ms
        note_duration: Optional[int] = None,  # in ms
    ) -> np.ndarray:
        """A messy way of generating a synthesized note"""
        velocity = np.random.randint(low=32, high=64)
        prompt_leading_silence = leading_silence_duration is None
        prompt_note_dur = note_duration is None
        
        if prompt_leading_silence or prompt_note_dur:
            leading_silence_duration, note_duration = self.get_random_duration()
        last_note_off_duration = int(self.max_duration*1000) - leading_silence_duration - note_duration
        
        note_idx = self.midi_notes.get_midi_idx(note, note_position)
        
        # initialize track + fluidsynth
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        
        track.append(mido.Message('program_change', program=program.value, time=0))
        track.append(mido.Message('note_on', note=note_idx, velocity=velocity, time=leading_silence_duration))
        track.append(mido.Message('note_off', note=note_idx, velocity=velocity, time=note_duration))
        track.append(mido.Message('note_on', note=0, velocity=velocity, time=last_note_off_duration))  # silence
        track.append(mido.Message('note_off', note=0, velocity=velocity, time=0))
        
        # use hash of (UNIX time + note + program) as file name to prevent collision
        f_name = f"{note.value}{note_position}_{program.name}_"
        f_name = f_name + MidiGenerator.apply_sha256(str(time.time()).replace(".", "") + str(note.value) + str(program.value))
        save_path = f"{self.tmp_path}/{f_name}."
        midi_path = save_path + "mid"
        wav_path = save_path + "wav"
        
        # save raw audio
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        midi.save(midi_path)  # save as .mid
        self._midi_to_wav(midi_path, wav_path)

        if not os.path.exists(wav_path):
            self._midi_to_wav(midi_path, wav_path)
        
        # reload and preprocess
        wav = self.post_process_wav(wav_path)  # postprocess wavfile
        
        # clear saved midi + wav
        self.__clear_tmp(save_path)
        return wav

    def _process_item(self, item: Tuple[MidiNote, MidiProgram, int, int, str]) -> None:
        note, program, position_idx, i, save_root = item
        wav = self.generate_midi_note(
            note=note,
            program=program,
            note_position=position_idx
        )
        save_path = f"{save_root}/wav/{note.value}{position_idx}_{program.name}_{i}.wav"
        label_path = f"{save_root}/labels.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.save_wav(wav, save_path)

        with open(label_path, "a") as f:
            f.write(f"{save_path},{len(wav)},{program.name},{note.value}{position_idx},{note.value}\n")
    
    def generate_dataset(
        self,
        cfg: DictConfig,
    ) -> pd.DataFrame:
        save_root = cfg.get("dataset_path", os.path.expanduser("~/dataset"))
        num_workers = cfg.get("num_workers", 0)
        samples_per_program_note = cfg.samples_per_program_note
        available_notes = [MidiNote(note) for note in cfg.available_notes]
        available_program = [MidiProgram(program) for program in cfg.available_program]

        # verbose
        logging.info("Generating dataset with the following configs:")
        logging.info(f"num_workers:              {num_workers}")
        logging.info(f"samples_per_program_note: {samples_per_program_note}")
        logging.info(f"available_notes:          {available_notes}")
        logging.info(f"available_program:        {available_program}")

        if num_workers < 0 or num_workers > os.cpu_count():
            num_workers = os.cpu_count()

        # initialize labels
        label_path = f"{save_root}/labels.csv"
        with open(label_path, "w") as f:
            f.write(f"wav_path,duration,instrument,note,label\n")
        
        # initialize spaces
        iter_spaces = []
        for note in available_notes:
            for program in available_program:
                note_positions = self.midi_notes.note_mapper[note.value]
                for position_idx in range(len(note_positions)):
                    for i in range(samples_per_program_note):
                        iter_spaces.append(
                            (note, program, position_idx, i, save_root)
                        )

        logging.info(f"Starting generation...")
        start_time = time.time()
        if num_workers == 0:
            # single thread
            for item in tqdm(iter_spaces, total=len(iter_spaces)):
                self._process_item(item)
        else:
            # multithread
            pool = mp.Pool(num_workers)
            pool.map(self._process_item, iter_spaces)
        
        labels = pd.read_csv(label_path).drop_duplicates()
        labels.to_csv(label_path, index=False)
        total_duration = MidiGenerator.sec_to_hour(labels["duration"].sum() / self.sampling_rate)
        elapsed_time = time.time() - start_time
        logging.info(f"Finished generated dataset with a total of {len(labels)} samples.")
        logging.info(f"Time took: {elapsed_time:.4f} seconds.")
        logging.info(f"A total of {total_duration} worth of dataset is created.")
        return labels
