import yaml


def load_yaml(filepath):
    with filepath.open('r') as file:
        return yaml.safe_load(file)
