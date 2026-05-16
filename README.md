# A Galaxy Morphology Classifier

A personal project to create a fairly straightforward ML package to classify galaxies morphologies. Doing this as execise in building something from start to finish.

[![CI](https://github.com/ZacharyTu42/galaxy-morph/actions/workflows/ci.yml/badge.svg)](https://github.com/ZacharyTu42/galaxy-morph/actions/workflows/ci.yml)

_Figure coming soon_

---

## Quickstart

> Once the package and model are released, this section will let you go from clone to prediction.

```bash
# Clone and install
git clone https://github.com/zacharytu42/galaxy-morph.git
cd galaxy-morph
conda env create -f environment.yml
conda activate galaxy-morph
make install # (or pip install -e .)

# Download the dataset (~2.5 GB)
make download-galaxy10

# Train the baseline model
make train

# Run inference on a single galaxy by sky coordinates
galaxy-morph predict --ra 184.74 --dec 47.30
```

---
