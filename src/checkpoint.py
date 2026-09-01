from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    encoder,
    decoder,
    optimizer,
    epoch: int,
    train_loss: float,
    best_val_loss: float,
    accelerator=None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if accelerator is not None:
        encoder_state_dict = accelerator.get_state_dict(encoder)
        decoder_state_dict = accelerator.get_state_dict(decoder)
    else:
        encoder_state_dict = encoder.state_dict()
        decoder_state_dict = decoder.state_dict()

    checkpoint_state = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "best_val_loss": float(best_val_loss),
        "encoder_state_dict": encoder_state_dict,
        "decoder_state_dict": decoder_state_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }

    torch.save(checkpoint_state, path)
    return path


def load_checkpoint(
    path: str | Path,
    encoder,
    decoder,
    optimizer=None,
    device: str | torch.device = "cpu",
):
    checkpoint = torch.load(Path(path), map_location=device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    train_loss = float(checkpoint.get("train_loss", 0.0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return epoch, train_loss, best_val_loss
