import os
from pathlib import Path

import soundfile as sf
from torch.utils.data import DataLoader

from src.dataset.dataset import VocoMorphDataset
from src.utils.parsers import parse_yaml


def save_random_batch_from_loader(dataset, sr: int, out_dir: Path, batch_size: int = 4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(loader))

    item_ids, effect_ids, raw_chunks, target_chunks = batch
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(item_ids)):
        raw = raw_chunks[i].squeeze().cpu().numpy()  # (time,)
        target = target_chunks[i].squeeze().cpu().numpy()

        sf.write(out_dir / f"{item_ids[i]}_raw.wav", raw.T, sr)
        sf.write(
            out_dir / f"{item_ids[i]}_effect{effect_ids[i].item()}.wav", target.T, sr
        )

    print(f"Saved {len(item_ids)} pairs to {out_dir}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
    DATA_ROOT = Path(os.environ["DATA_ROOT"])

    model_dir = PROJECT_ROOT.joinpath("models", "VocoMorphUnAtt")
    assert model_dir.exists(), f"{model_dir} does not exist"

    yaml_path = model_dir.joinpath("config.yaml")
    yaml_dict = parse_yaml(yaml_path)
    config = yaml_dict["config"]["data"]

    dataset = VocoMorphDataset(
        config,
        DATA_ROOT.joinpath("timit", "datalists", "train_augmented.csv"),
    )
    save_random_batch_from_loader(
        dataset,
        sr=config["sample_rate"],
        out_dir=DATA_ROOT.joinpath("output", "test_dataloader"),
    )
