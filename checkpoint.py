import os

import torch

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "mae.pt")
MODEL_CONFIG_KEYS = ("image_size", "patch_size", "n", "len_keep")


def _config_compatible(saved, current):
    for key in MODEL_CONFIG_KEYS:
        if saved.get(key) != current.get(key):
            return False, key
    return True, None


def save_checkpoint(
    path,
    model,
    optimizer,
    global_step,
    loss_history,
    config,
    image_idx,
    step_in_image,
    images_completed,
    image_path,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "loss_history": loss_history,
            "config": config,
            "image_idx": image_idx,
            "step_in_image": step_in_image,
            "images_completed": images_completed,
            "image_path": image_path,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, config, device):
    if not os.path.exists(path):
        return 0, [], None, 0, 0

    ckpt = torch.load(path, map_location=device, weights_only=False)
    ok, bad_key = _config_compatible(ckpt.get("config", {}), config)
    if not ok:
        raise ValueError(
            f"checkpoint config mismatch on {bad_key!r}: "
            f"saved {ckpt.get('config')} vs current {config}"
        )

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if "image_idx" not in ckpt and "epoch" in ckpt:
        print("warning: old epoch checkpoint — starting a fresh random image block")
        return ckpt["global_step"], ckpt.get("loss_history", []), None, 0, ckpt.get("epoch", 0)

    return (
        ckpt["global_step"],
        ckpt.get("loss_history", []),
        ckpt.get("image_idx"),
        ckpt.get("step_in_image", 0),
        ckpt.get("images_completed", 0),
    )
