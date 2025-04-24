import os
import sys

if os.environ.get("PROJECT_ROOT", None) is None:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    os.environ["PROJECT_ROOT"] = PROJECT_ROOT
    sys.path.append(PROJECT_ROOT)  # ensure src/ is accessible

import argparse
from importlib import import_module
from src.utils import parse_yaml


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
        "--sample-file",
        type=str,
        default=None,
        help="Path to the sample audio file for inference.",
    )

    parser.add_argument(
        "--out-wav-dir",
        type=str,
        default=None,
        help="(Optional) directory to save output WAV files in test mode.",
    )

    args = parser.parse_args()

    if args.mode == "infer_sample" and args.sample_file is None:
        parser.error("--sample-file is required when --mode is set to 'infer_sample'.")

    return args


if __name__ == "__main__":
    args = parse_args()

    model_dir = os.path.join(os.environ["PROJECT_ROOT"], "models", args.model_name)
    assert os.path.exists(model_dir), f"{model_dir} does not exist"
    yaml_path = os.path.join(model_dir, "config.yaml")
    yaml_dict = parse_yaml(yaml_path)
    config = yaml_dict["config"]

    if args.generate_dataset:
        module = f"src.dataset.download_dataset"
        module = import_module(module)
        module.main(args.generate_dataset)
        module = f"src.dataset.generate_dataset"
        module = import_module(module)
        module.main(args.generate_dataset, config["data"])
        sys.exit(0)

    # validate arguments
    if args.run_tests:
        module = import_module("tests.main")
        module.main(config["data"])
        sys.exit(0)

    if args.mode:
        # import and run main function from trainer
        module = import_module("src.main")
        module.main(args, config)
