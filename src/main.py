import os
from argparse import Namespace

from models.factory import create_model_instance
from .dataset.dataset import get_dataloaders
from .utils import get_device, set_seed
from .trainer.trainer import ModelTrainer


def main(args: Namespace, config: dict):

    model_dir = os.path.join(os.environ["PROJECT_ROOT"], "models", args.model_name)
    device = get_device(args.device)

    model = create_model_instance(args.model_name, config["model"])
    trainer = ModelTrainer(config, model_dir, model, device)

    if args.mode == "train":
        set_seed(config["seed"])
        loaders = get_dataloaders(["train", "valid"], config["data"])
        epochs = config["trainer"]["max_epoch"]
        trainer.train(epochs, loaders["train"], loaders["valid"])
    elif args.mode == "test":
        loaders = get_dataloaders(["test"], config["data"])
        trainer.test(loaders["test"], args.out_wav_dir)
    elif args.mode == "infer_sample":
        # TODO: implement infer_sample mode
        pass
    else:
        # TODO: live inference
        pass
