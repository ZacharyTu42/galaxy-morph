import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from galaxy_morph.data.datasets import Galaxy10Dataset

app = typer.Typer()


@app.command()
def compute_stats(h5_path: str, batch_size: int = 64, num_workers: int = 4) -> None:
    """Compute per-channel mean and std for Galaxy10 DECaLS dataset."""

    dataset = Galaxy10Dataset(h5_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels = 0

    for images, _ in tqdm(loader, desc="Computing normalization stats"):
        # images shape: (B, C, H, W)
        channel_sum += images.sum(dim=[0, 2, 3])  # Sum over batch and spatial dims
        channel_sum_sq += (images**2).sum(dim=[0, 2, 3])
        num_pixels += images.numel() // 3  # Total pixels per channel

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sum_sq / num_pixels - mean**2)

    print(f"Channel means: {mean.tolist()}")
    print(f"Channel stds: {std.tolist()}")


if __name__ == "__main__":
    app()
