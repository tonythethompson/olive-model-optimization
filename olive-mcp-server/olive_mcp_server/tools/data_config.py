"""Tool: get_data_config_template."""

from typing import Any


def get_data_config_template(
    data_source: str,
    task: str = "calibration",
) -> dict[str, Any]:
    """Generate a data pipeline configuration for Olive.

    Args:
        data_source: e.g. "huggingface", "local_files", "image_folder".
        task: "calibration" or "evaluation".

    Returns:
        Complete DataConfig JSON with preprocessing, sampling, and batching.
    """
    source = data_source.lower().strip()
    task = task.lower().strip()

    if source == "huggingface":
        container_type = "HuggingFaceContainer"
        load_cfg = {
            "path": "<dataset_name>",
            "split": "train" if task == "calibration" else "validation",
            "subset": "",
        }
        pre_process = {
            "input_cols": ["text"],
            "label_cols": ["label"],
            "padding": "max_length",
            "max_length": 512,
            "normalization": "none",
        }
    elif source == "image_folder":
        container_type = "ImageNetContainer"
        load_cfg = {
            "data_dir": "<path/to/image_folder>",
        }
        pre_process = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "input_size": [224, 224],
            "interpolation": "bilinear",
            "normalization": "ImageNet",
        }
    else:
        container_type = "DataContainer"
        load_cfg = {
            "data_dir": "<path/to/data>",
            "data_files": ["train.json"],
        }
        pre_process = {
            "input_cols": ["input"],
            "label_cols": ["label"],
            "normalization": "custom",
        }

    sampling_size = 100 if task == "calibration" else 1000

    # Build a proper Olive data_configs entry
    data_config_entry = {
        "name": f"{task}_data",
        "type": container_type,
        "load_dataset_config": load_cfg,
        "pre_process_data_config": pre_process,
        "dataloader_config": {
            "batch_size": 1,
            "drop_last": False,
            "num_workers": 0,
        },
        "post_process_data_config": {
            "output_cols": ["output"],
        },
    }

    if task == "calibration":
        data_config_entry["sampling"] = sampling_size

    return {
        "data_configs": [data_config_entry],
        "notes": [
            "Calibration data must match the inference distribution.",
            "For static quantization, 100-300 samples are usually sufficient.",
            "ImageNet normalization is [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225].",
        ],
    }
