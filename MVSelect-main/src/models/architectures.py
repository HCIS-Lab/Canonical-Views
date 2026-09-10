"""Shared model-architecture names and lightweight inference helpers."""

import os
import re


SUPPORTED_ARCHITECTURES = ("resnet18", "vit", "tinyvit")
ARCHITECTURE_CHOICES = ("auto",) + SUPPORTED_ARCHITECTURES
ARCHITECTURE_FEATURE_DIMS = {
    "resnet18": 512,
    "vit": 768,
    "tinyvit": 320,
}
TINYVIT_TIMM_MODEL = "tiny_vit_5m_224.dist_in22k_ft_in1k"


def infer_architecture_from_name(path_or_name):
    """Infer the architecture prefix from an experiment or checkpoint path."""
    if not path_or_name:
        return None
    text = os.path.basename(os.path.normpath(str(path_or_name)))
    text = re.sub(r"^freeze_\d+_", "", text)
    for arch in SUPPORTED_ARCHITECTURES:
        if text.startswith(f"{arch}steps") or text.startswith(f"{arch}train"):
            return arch

    # A checkpoint path normally has the architecture in its parent run name.
    parent = os.path.basename(os.path.dirname(os.path.normpath(str(path_or_name))))
    parent = re.sub(r"^freeze_\d+_", "", parent)
    for arch in SUPPORTED_ARCHITECTURES:
        if parent.startswith(f"{arch}steps") or parent.startswith(f"{arch}train"):
            return arch
    return None


def infer_architecture_from_state(state):
    """Infer architecture from the classifier input width in a state dict."""
    if not isinstance(state, dict):
        return None
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    weight = state.get("classifier.weight")
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    feature_dim = int(shape[1])
    matches = [
        arch for arch, dim in ARCHITECTURE_FEATURE_DIMS.items()
        if dim == feature_dim
    ]
    return matches[0] if len(matches) == 1 else None


def resolve_architecture(requested, path_or_name=None, state=None):
    """Resolve ``auto`` and reject path/checkpoint architecture mismatches."""
    if requested not in ARCHITECTURE_CHOICES:
        raise ValueError(
            f"Unsupported architecture '{requested}'. "
            f"Choose from {', '.join(ARCHITECTURE_CHOICES)}."
        )

    from_name = infer_architecture_from_name(path_or_name)
    from_state = infer_architecture_from_state(state)
    inferred = from_name or from_state
    if from_name and from_state and from_name != from_state:
        raise ValueError(
            f"Experiment path implies '{from_name}', but the checkpoint "
            f"classifier implies '{from_state}'."
        )

    if requested == "auto":
        if inferred is None:
            raise ValueError(
                "Could not infer architecture from the experiment name or "
                "checkpoint. Pass --arch explicitly."
            )
        return inferred

    if inferred and requested != inferred:
        raise ValueError(
            f"--arch {requested} does not match the inferred checkpoint "
            f"architecture '{inferred}'."
        )
    return requested
