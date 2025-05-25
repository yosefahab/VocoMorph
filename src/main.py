import os
from argparse import Namespace

import torch

from models.factory import create_model_instance
from .infer import infer
from .dataset.dataset import get_dataloaders
from .utils import get_device, set_seed
from .trainer.trainer import ModelTrainer


def main(args: Namespace, config: dict):

    model_dir = os.path.join(os.environ["PROJECT_ROOT"], "models", args.model_name)
    device = get_device(args.device)

    model = create_model_instance(args.model_name, config["model"])

    if args.mode == "train":
        trainer = ModelTrainer(config, model_dir, model, device)
        set_seed(config["seed"])
        loaders = get_dataloaders(["train", "valid"], config["data"])
        epochs = config["trainer"]["max_epoch"]
        trainer.train(epochs, loaders["train"], loaders["valid"])
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
        infer(2, args.sample_file, model, config)
    else:
        # TODO: live inference
        pass
