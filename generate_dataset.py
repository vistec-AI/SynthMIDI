import hydra
from omegaconf import DictConfig, OmegaConf

from midisynth.dataset.midi import MidiGenerator
from midisynth.dataset.enum import MidiNote, MidiProgram


@hydra.main(version_base=None, config_path="conf", config_name="dataset")
def main(cfg: DictConfig) -> None:   
    midi_generator = MidiGenerator(sampling_rate=cfg.sampling_rate)
    midi_generator.generate_dataset(cfg)


if __name__ == "__main__":
    main()
