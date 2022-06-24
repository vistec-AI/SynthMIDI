import hydra
from omegaconf import DictConfig, OmegaConf

from midi import MidiGenerator
from midi_enum import MidiNote, MidiProgram


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:   
    midi_generator = MidiGenerator(sampling_rate=cfg.sampling_rate)
    midi_generator.generate_dataset(cfg)


if __name__ == "__main__":
    main()
