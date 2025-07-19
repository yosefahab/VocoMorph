import os
import sys
from pathlib import Path

if os.environ.get("PROJECT_ROOT", None) is None:
    PROJECT_ROOT = Path(__file__).parent.absolute()
    os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
    sys.path.append(str(PROJECT_ROOT))  # ensure src/ is accessible

if os.environ.get("DATA_ROOT", None) is None:
    DATA_ROOT = Path(os.environ["PROJECT_ROOT"]).joinpath("data")
    os.environ["DATA_ROOT"] = str(DATA_ROOT)

import argparse
from importlib import import_module

from src.utils.parsers import parse_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train, test, or infer using a model configured with a .yaml file",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--generate-dataset",
        choices=["librispeech", "timit"],
        nargs="?",
        const="timit",
        help="Download a dataset. Default: timit.",
    )

    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the test scripts (not the model).",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="VocoMorphBase",
        help="Name of the model to use.",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Specify the device to use: 'cpu' or 'gpu'.",
    )

    parser.add_argument(
        "--mode",
        choices=["train", "test", "infer_sample"],
        default=None,
        help="Specify mode: 'train' (training), 'test' (evaluation), or 'infer_sample' (inference on a sample).",
    )
    parser.add_argument(
        "--out-wav-dir",
        type=str,
        default=None,
        help="(Optional) directory to save output WAV files in test mode.",
    )

    parser.add_argument(
        "--sample-file",
        type=str,
        default=None,
        help="Path to the sample audio file for inference.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="(Optional) checkpoint to use in inference mode.",
    )

    args = parser.parse_args()

    if args.mode == "infer_sample" and (args.sample_file is None):
        parser.error("--sample-file is required when --mode is set to 'infer_sample'.")

    if args.mode == "infer_sample" and (args.checkpoint_path is None):
        parser.error(
            "--checkpoint-path is required when --mode is set to 'infer_sample'."
        )
    return args


if __name__ == "__main__":
    args = parse_args()

    model_dir = Path(os.environ["PROJECT_ROOT"]).joinpath("models", args.model_name)
    assert model_dir.exists(), f"{model_dir} does not exist"
    yaml_path = model_dir.joinpath("config.yaml")
    yaml_dict = parse_yaml(yaml_path)
    config = yaml_dict["config"]

    if args.generate_dataset:
        module = "src.dataset.download_dataset"
        module = import_module(module)
        module.main(args.generate_dataset)
        module = "src.dataset.generate_dataset"
        module = import_module(module)
        module.main(args.generate_dataset, config["data"])
        exit(0)

    # validate arguments
    if args.run_tests:
        module = import_module("tests.main")
        module.main(config["data"])
        exit(0)

    if args.mode:
        # import and run main function from trainer
        module = import_module("src.main")
        module.main(args, config)
