import torch.nn as nn


ACTIVATION_MODES = ("first-on", "all-on", "all-off")
ACTIVATION_MODE_ALIASES = {
    "first-on": "first-on",
    "first-only": "first-on",
    "all-on": "all-on",
    "all": "all-on",
    "all-off": "all-off",
    "none": "all-off",
}
ACTIVATION_MODE_CHOICES = tuple(ACTIVATION_MODE_ALIASES)
ACTIVATION_KINDS = ("prelu", "silu", "gelu")


def normalize_activation_mode(mode: str | None) -> str:
    """Return the canonical activation placement mode."""

    if mode is None:
        return "first-on"
    key = str(mode).strip().lower()
    if key not in ACTIVATION_MODE_ALIASES:
        raise ValueError(
            f"activation_mode must be one of {ACTIVATION_MODE_CHOICES}, got {mode!r}."
        )
    return ACTIVATION_MODE_ALIASES[key]


def activation_flags(mode: str | None, num_layers: int) -> list[bool]:
    """Return per-layer activation flags for a canonical or alias placement mode."""

    if num_layers < 0:
        raise ValueError("num_layers must be non-negative.")
    canonical = normalize_activation_mode(mode)
    if canonical == "all-on":
        return [True] * num_layers
    if canonical == "all-off":
        return [False] * num_layers
    return [idx == 0 for idx in range(num_layers)]


def build_activation(kind: str, width: int) -> nn.Module:
    """Construct an activation module for a hidden layer of ``width`` channels."""

    key = str(kind).strip().lower()
    if key == "prelu":
        return nn.PReLU(num_parameters=int(width))
    if key == "silu":
        return nn.SiLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"activation_kind must be one of {ACTIVATION_KINDS}, got {kind!r}.")
