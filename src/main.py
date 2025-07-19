import os
from argparse import Namespace
from pathlib import Path

import torch

from models.factory import create_model_instance
from src.dataset.dataset import get_dataloaders
from src.infer import infer
from src.trainer.trainer import ModelTrainer
from src.utils.device import get_device
from src.utils.seed import set_seed

assert os.environ.get("PROJECT_ROOT") is not None


def main(args: Namespace, config: dict):
    model_dir = Path(os.environ["PROJECT_ROOT"]).joinpath("models", args.model_name)
    device = get_device(args.device)

    model = create_model_instance(args.model_name, config["model"])

    if args.mode == "train":
        trainer = ModelTrainer(config, model_dir, model, device)
        set_seed(config["seed"])
        loaders = get_dataloaders(["train", "valid", "test"], config["data"])
        epochs = config["trainer"]["max_epoch"]
        trainer.train(epochs, loaders["train"], loaders["valid"], loaders["test"])
    elif args.mode == "test":
        trainer = ModelTrainer(config, model_dir, model, device)
        loaders = get_dataloaders(["test"], config["data"])
        trainer.test(loaders["test"], args.out_wav_dir)
    elif args.mode == "infer_sample":
        checkpoint = torch.load(
            args.checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        output_filename = Path(args.sample_file).stem

        assert os.environ.get("DATA_ROOT") is not None
        output_dir = Path(os.environ["DATA_ROOT"]).joinpath("output", output_filename)
        for i in range(len(config["data"]["effects"])):
            infer(i, args.sample_file, model, config, device, output_path=output_dir)
    else:
        # TODO: live inference
        pass
