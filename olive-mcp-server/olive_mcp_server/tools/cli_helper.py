"""Tool: get_cli_command."""

from typing import Any


def get_cli_command(
    optimization_goal: str,
    model: str,
    target: str = "cpu",
    config_path: str = "olive_config.json",
    output_dir: str = "./models/optimized",
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Generate a ready-to-run Olive CLI command.

    Args:
        optimization_goal: "quantize", "finetune", "optimize", "auto-opt",
            "onnx-graph-capture", or "generate-adapter".
        model: Model name, path, or HuggingFace ID.
        target: Target accelerator, e.g. "cpu", "gpu", "npu", or "tensorrt".
        config_path: Path to workflow JSON config.
        output_dir: Output directory.
        batch_size: Optional batch size to append.

    Returns:
        CLI command string and flag explanation.
    """
    goal = optimization_goal.lower().strip()
    accepted = {
        "quantize": "olive quantize",
        "finetune": "olive finetune",
        "optimize": "olive optimize",
        "auto-opt": "olive auto-opt",
        "onnx-graph-capture": "olive onnx-graph-capture",
        "generate-adapter": "olive generate-adapter",
    }
    if goal not in accepted:
        raise ValueError(f"Unrecognized optimization_goal '{goal}'. Must be one of: {', '.join(accepted.keys())}")
    base = accepted[goal]

    flags = [
        f'--config "{config_path}"',
        f'--model-path "{model}"',
        f'--output-dir "{output_dir}"',
        f"--accelerator {target}",
    ]
    if batch_size is not None:
        flags.append(f"--batch-size {batch_size}")

    command = f"{base} " + " ".join(flags)

    explanations = {
        "--config": "Path to the Olive workflow configuration JSON.",
        "--model-path": "Input model path or HuggingFace model ID.",
        "--output-dir": "Directory for the optimized model artifacts.",
        "--accelerator": "Target accelerator (cpu, gpu, npu, tensorrt).",
        "--batch-size": "Inference batch size for profiling or optimization.",
    }

    return {
        "command": command,
        "optimization_goal": goal,
        "flag_explanations": explanations,
        "notes": [
            "Ensure Olive CLI is installed: pip install olive-ai",
            "If using a config file, most flags can be omitted.",
            "For batch inference, increase --batch-size and set dynamic_axes in OnnxConversion.",
        ],
    }
