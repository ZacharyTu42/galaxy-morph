# Data Directory

This directory is intentionally mostly empty in the repository. Dataset files are gitignored and must be downloaded locally.

## Layout

```
data/
├── raw/          # original, unmodified source files (downloaded from Zenodo)
└── processed/    # derived files produced by preprocessing scripts
```

## Downloading Galaxy10 DECaLS

The dataset (~2.5 GB HDF5 file) is hosted on Zenodo. Download it with the CLI:

```bash
galaxy-morph download
```

This places `Galaxy10_DECals.h5` in `data/raw/` and verifies its SHA-256 checksum automatically. The download is idempotent — re-running it does nothing if the file is already present and the checksum matches.

To force a redownload:

```bash
galaxy-morph download --force
```

To download to a different location:

```bash
galaxy-morph download --dest /path/to/dir
```

## Source

- **Dataset:** Galaxy10 DECaLS
- **Zenodo record:** <https://zenodo.org/records/10845026>
- **Classes:** 10 morphological categories (smooth, featured disk, edge-on, etc.)
- **Size:** ~17,000 images at 256×256 px, RGB
