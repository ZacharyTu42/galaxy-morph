from pathlib import Path

import typer

from .data.download import download_galaxy10

app = typer.Typer()


@app.command()
def download(dest: Path = Path("data/raw"), force: bool = False) -> None:
    """Download the Galaxy10 DECaLS dataset from Zenodo."""
    try:
        file_path = download_galaxy10(dest, force)
        typer.echo(f"Downloaded Galaxy10 DECaLS dataset to {file_path}")
    except Exception as e:
        typer.echo(f"Error downloading dataset: {e}", err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
