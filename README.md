# DML-4-Fluxes-CHM

Hybrid modeling approaches for flux partitioning from the ["Causal hybrid modeling with double machine learning—applications in carbon flux modeling"](https://iopscience.iop.org/article/10.1088/2632-2153/ad5a60) manuscript.

## Data Source

The model requires FLUXNET2015 half-hourly Fullset data. You can download this data from the [FLUXNET2015 Dataset](https://fluxnet.org/data/fluxnet2015-dataset/). After downloading, store the CSV files in the `data/` directory.

## Installation

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate dml4fluxes
```

2. Install the package:
```bash
pip install -e .
```

## Usage

To perform flux partitioning, run:
```bash
python scripts/partition.py
```

This will process the FLUXNET2015 data and generate the flux partitioning results.

## License

This project is licensed under the terms included in the LICENSE file.