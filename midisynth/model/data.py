from enum import Enum
from typing import Dict, Tuple

import pandas as pd
import torchaudio
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchaudio.compliance import kaldi

from ..dataset.enum import MidiNote


class SpeechFeature(Enum):
    FBANK = "fbank"


def get_class_mapper() -> Dict[str, int]:
    return {
        note.value: i
        for i, note in enumerate(list(MidiNote))
    }


class SynthMidiDataset(Dataset):

    def __init__(self, csv_path: str, feature_cfg: DictConfig) -> None:
        self.data_root = feature_cfg.get("data_root", "midisynth_dataset")
        self.feat_name = SpeechFeature(feature_cfg.get("feat_name", "fbank"))
        self.feat_param = feature_cfg.get("feat_param", {})
        self.class_mapper = get_class_mapper()

        self.csv_path = csv_path
        self.labels = pd.read_csv(self.csv_path)

        torchaudio.set_audio_backend(feature_cfg.torchaudio_backend)  # for macOS/linux

    def __len__(self) -> int:
        return len(self.labels)

    def extract_feature(
        self, 
        wav: Tensor, 
        sampling_rate: int,
    ) -> Tensor:
        if self.feat_name.value == SpeechFeature.FBANK.value:
            return kaldi.fbank(
                wav,
                sample_frequency=sampling_rate,
                **self.feat_param
            )
        else:
            raise NameError(f"Unrecognize feature name: {self.feat_name}")

    def __getitem__(self, index: int) -> Tuple[int, int]:
        sample = self.labels.iloc[index]
        label = self.class_mapper[sample["label"]]
        wav_path = sample["wav_path"]
        wav_path = f"{self.data_root}/wav/{wav_path}"
        if ".wav" not in wav_path:
            wav_path += ".wav"

        wav, sr = torchaudio.load(wav_path)
        feature = self.extract_feature(wav, sampling_rate=sr)

        return feature, label
