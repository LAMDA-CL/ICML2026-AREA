import json
from pathlib import Path


def load_class_names(dataset_name):
    labels_path = Path(__file__).resolve().with_name("labels.json")
    with labels_path.open(encoding="utf-8") as labels_file:
        labels = json.load(labels_file)
    if dataset_name not in labels:
        raise ValueError("No class names in labels.json for {!r}".format(dataset_name))
    return labels[dataset_name]


def normalize_class_name(class_name):
    return class_name.replace(" ", "_").replace("/", "_")


def get_description_path(directory, class_name, class_index):
    directory = Path(directory)
    filename = normalize_class_name(class_name) + "_descriptions.txt"
    candidates = [directory / filename,
                  directory / "{:03d}.{}".format(class_index + 1, filename)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Missing descriptions for {!r}; checked: {}".format(
            class_name, ", ".join(str(candidate) for candidate in candidates)))
