import torch
from typing import Union

__all__ = ["load_policy"]


def _extract_actor(checkpoint: Union[list, dict, torch.nn.Module]):
    """Return the actor network from various checkpoint formats.

    The SHAC algorithm saves a *list* ``[actor, critic, target_critic, ...]``.
    Users might also provide a plain *state dict* or a scripted module. We try to
    handle the most common cases gracefully.
    """
    if isinstance(checkpoint, (list, tuple)) and len(checkpoint) >= 1:
        return checkpoint[0]

    # If the checkpoint is a dict try typical keys
    if isinstance(checkpoint, dict):
        for key in ("actor", "policy", "model", "policy_state_dict"):
            if key in checkpoint:
                return checkpoint[key]

    # Fallback – maybe the checkpoint itself is the actor module
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    raise RuntimeError(
        "Could not infer actor network from the provided checkpoint file – please check its format."
    )


def load_policy(path: str, device: str = "cpu") -> torch.nn.Module:
    """Load an actor network saved with SHAC (or similar) and move to *device*."""
    # First try regular torch.load (supports Python pickled modules, list, dict, …)
    checkpoint = torch.load(path, map_location=device)

    try:
        actor = _extract_actor(checkpoint)
    except RuntimeError:
        # Fallback: maybe the file is a TorchScript module (torch.jit.save)
        try:
            actor = torch.jit.load(path, map_location=device)
        except Exception as e:
            raise RuntimeError(
                "Could not load an actor network from the given checkpoint. "
                "If you saved only the model's `state_dict`, you must recreate "
                "the network architecture and load the state dict yourself."
            ) from e

    # If we ended up with a plain state_dict, we cannot proceed automatically
    if isinstance(actor, dict):
        raise RuntimeError(
            "Checkpoint contains only a parameter state_dict – the network "
            "architecture is unknown. Please provide a checkpoint that "
            "includes the full Actor module (as saved by SHAC's save() method "
            "or a TorchScript module)."
        )

    actor = actor.to(device)
    actor.eval()
    return actor
