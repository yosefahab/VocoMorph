import yaml
from pathlib import Path


def parse_yaml(yaml_path: Path) -> dict:
    """
    Parse and return the contents of a YAML file.
    Args:
    - path: Path to the YAML file to be parsed.
    Returns:
        dict: A dictionary containing the parsed contents of the YAML file.
    """
    assert yaml_path.exists(), f"YAML file {yaml_path} doesn't exist"
    with open(yaml_path, "r") as yaml_file:
        config_dict = yaml.full_load(yaml_file)
        return config_dict
