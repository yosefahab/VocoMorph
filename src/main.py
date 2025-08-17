import os
from argparse import Namespace
from pathlib import Path


from models.factory import create_model_instance
from src.dataset.dataset import get_dataloaders
from src.infer import infer
from src.trainer.checkpointer import Checkpointer
from src.trainer.strategy import DDPStrategy, SingleDeviceStrategy
from src.trainer.trainer import ModelTrainer
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.types import DictConfig

assert os.environ.get("PROJECT_ROOT") is not None


def main(args: Namespace, config: DictConfig):
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
    DATA_ROOT = Path(os.environ["DATA_ROOT"])

    model_dir = PROJECT_ROOT.joinpath("models", args.model)
    model = create_model_instance(model_dir, config["model"])

    device_type = args.device
    if args.ddp:
        assert "LOCAL_RANK" in os.environ
    args.ddp = args.ddp or "LOCAL_RANK" in os.environ or "RANK" in os.environ

    if args.mode == "train":
        set_seed(config["seed"])
        strategy = (
            DDPStrategy(device_type) if args.ddp else SingleDeviceStrategy(device_type)
        )
        trainer = ModelTrainer(config, model_dir, model, strategy, device_type)
        splits = ["train", "valid", "test"]
        dataset_path = DATA_ROOT.joinpath(args.dataset)
        loaders = get_dataloaders(dataset_path, splits, config["data"], args.ddp)
        epochs = config["trainer"]["max_epoch"]

        trainer.train(epochs, loaders["train"], loaders["valid"], loaders["test"])

    elif args.mode == "test":
        strategy = (
            DDPStrategy(device_type) if args.ddp else SingleDeviceStrategy(device_type)
        )
        trainer = ModelTrainer(config, model_dir, model, strategy, device_type)
        splits = ["test"]
        dataset_path = DATA_ROOT.joinpath(args.dataset)
        loaders = get_dataloaders(dataset_path, splits, config["data"], args.ddp)

        trainer.test(loaders["test"], args.out_wav_dir)

    elif args.mode == "infer_sample":
        data_root = Path(os.environ.get("DATA_ROOT", ""))
        assert data_root.exists()

        checkpoints_dir = model_dir.joinpath("checkpoints")
        assert checkpoints_dir.exists()

        import torch

        checkpoint = torch.load(
            args.checkpoint_path,
            map_location=get_device(device_type),
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        filepath = args.sample_file
        output_filename = Path(args.sample_file).stem
        output_path = data_root.joinpath("output", output_filename)
        output_path.mkdir(exist_ok=True)
        for effect_id in range(len(config["data"]["effects"])):
            infer(effect_id, filepath, model, config, device_type, output_path)

    else:
        # TODO: live inference
        pass
