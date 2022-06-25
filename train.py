import logging
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from midisynth.model.data import SynthMidiDataset
from midisynth.model import Baseline


@hydra.main(version_base=None, config_path="conf", config_name="baseline")
def main(cfg: DictConfig) -> None:
    logging.info("Generating Dataloader...")
    train_dataloader = DataLoader(
        SynthMidiDataset(
            cfg.data.train.csv_path,
            cfg.feature
        ),
        **cfg.data.train.dataloader
    )

    val_dataloader = DataLoader(
        SynthMidiDataset(
            cfg.data.val.csv_path,
            cfg.feature
        ),
        **cfg.data.val.dataloader
    )

    test_dataloader = DataLoader(
        SynthMidiDataset(
            cfg.data.test.csv_path,
            cfg.feature
        ),
        **cfg.data.test.dataloader
    )

    logging.info("Start Training!")
    model = Baseline(cfg.model)
    model.fit(
        train_dataloader,
        val_dataloader,
        **cfg.trainer
    )

    logging.info("Training Finished! Evaluating...")
    results = model.evaluate(test_dataloader)

    logging.info(results)


if __name__ == "__main__":
    main()
