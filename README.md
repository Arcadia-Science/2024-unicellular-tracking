# 2024-unicellular-tracking

[![run with conda](https://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/projects/miniconda/en/latest/)
[![Arcadia Pub](https://img.shields.io/badge/Arcadia-Pub-596F74.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDI3LjcuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCA0My4yIDQwLjQiIHN0eWxlPSJlbmFibGUtYmFja2dyb3VuZDpuZXcgMCAwIDQzLjIgNDAuNDsiIHhtbDpzcGFjZT0icHJlc2VydmUiPgo8c3R5bGUgdHlwZT0idGV4dC9jc3MiPgoJLnN0MHtmaWxsOm5vbmU7c3Ryb2tlOiNGRkZGRkY7c3Ryb2tlLXdpZHRoOjI7c3Ryb2tlLWxpbmVqb2luOmJldmVsO3N0cm9rZS1taXRlcmxpbWl0OjEwO30KPC9zdHlsZT4KPGc+Cgk8cG9seWdvbiBjbGFzcz0ic3QwIiBwb2ludHM9IjIxLjYsMyAxLjcsMzcuNCA0MS41LDM3LjQgCSIvPgoJPGxpbmUgY2xhc3M9InN0MCIgeDE9IjIxLjYiIHkxPSIzIiB4Mj0iMjEuNiIgeTI9IjI3LjMiLz4KCTxwb2x5bGluZSBjbGFzcz0ic3QwIiBwb2ludHM9IjEyLjIsMTkuNCAyNC42LDMwLjEgMjQuNiwzNy40IAkiLz4KCTxsaW5lIGNsYXNzPSJzdDAiIHgxPSIxNy42IiB5MT0iMTYuNyIgeDI9IjE3LjYiIHkyPSIyNC4xIi8+Cgk8bGluZSBjbGFzcz0ic3QwIiB4MT0iMjguNiIgeTE9IjE1LjIiIHgyPSIyMS43IiB5Mj0iMjIuMSIvPgoJPHBvbHlsaW5lIGNsYXNzPSJzdDAiIHBvaW50cz0iNi44LDI4LjcgMTkuNSwzNC40IDE5LjUsMzcuNCAJIi8+Cgk8bGluZSBjbGFzcz0ic3QwIiB4MT0iMzQuOCIgeTE9IjI1LjgiIHgyPSIyNC42IiB5Mj0iMzYuMSIvPgoJPGxpbmUgY2xhc3M9InN0MCIgeDE9IjI5LjciIHkxPSIyMi4yIiB4Mj0iMjkuNyIgeTI9IjMwLjkiLz4KPC9nPgo8L3N2Zz4K)](https://doi.org/10.57844/arcadia-2d61-fb05)

![tracked cells](resources/cell-tracks-in-pools.gif)


## Purpose

This repository accompanies the pub "[A high-throughput imaging assay for phenotyping unicellular swimming](https://doi.org/10.57844/arcadia-2d61-fb05)". Its main purpose is for detecting and tracking unicellular organisms in brightfield time-lapse microscopy data at scale.


## Installation and setup

This repository uses conda to manage software environments and installations. If you do not already have conda installed, you can find operating system-specific instructions for installing miniconda [here](https://docs.anaconda.com/miniconda/). After installing conda, navigate to a directory where you would like to clone the repository to, and run the following commands to create the pipeline run environment.

```{bash}
git clone https://github.com/Arcadia-Science/2024-unicellular-tracking.git
conda env create -n unicellular-tracking --file envs/dev.yml
conda activate unicellular-tracking
pip install -e .
```

If the installation was successful, the below command will return without error.
```bash
python -c "import chlamytracker"
```


## Overview

### Description of the folder structure
This repository is organized into the following top-level directories.
* **btrack_config**: contains a YAML file for configuring [`btrack`](https://btrack.readthedocs.io/en/latest/index.html) configuration.
* **data**: CSV files containing summary motility metrics from measured cell trajectories.
* **envs**: contains a conda environment file that lists the packages and dependencies used for creating the conda environment.
* **notebooks**: Collection of Jupyter notebooks for analyzing motility data, including the code used to generate Figures 4–7 in the pub.
* **resources**: Static files such as PNGs and GIFs used for documentation within the repository.
* **results**: A collection of SVG files output by the Jupyter notebooks for generating Figures 4–7 in the pub.
* **src/chlamytracker**: Source code, scripts, and tests comprising the key functionality of the repository including parallelized image processing, cell tracking, and statistical analysis.

### Methods

#### Cell tracking
Cell tracking was performed by running the `track_cells.py` script (see the "Scripts" section below for more context) on the full dataset of raw brightfield microscopy time lapses available at https://doi.org/10.6019/S-BIAD1298. As described in [data/README.md](data/README.md), this dataset is comprised of _Chlamydomonas reinhardtii_ cells swimming in either agar microchamber pools (`AMID-04_CC-124_pools`) or microtiter plates (`AMID-05_CC-124_wells`). The following command was run to track cells in microchamber pools:
```bash
python src/chlamytracker/scripts/track_cells.py \
    AMID-04_CC-124_pools/S1-Cr3-T/ \
    --vessel "pools" \
    --pool-radius 50 \
    --use-dask
```
The same command was repeated for the next three subdirectories (`S2-Cr3-M`, `S3-Cr4-T`, and `S4-Cr4-M`) by substituting in the name of the subdirectory to the first argument. For tracking cells in microtiter plates, the same script was run with the following optional arguments,
```bash
python src/chlamytracker/scripts/track_cells.py \
    AMID-05_CC-124_wells/ \
    --vessel "384-well plate"
    --use-dask
```

#### Generating figures
The statistical analysis was done through a series of Jupyter notebooks in which the figures of the pub were also created. The list below maps each analysis and figure to its notebook.
* Figure 4: [`2_temporal-variation-in-motility-metrics.ipynb`](notebooks/2_temporal-variation-in-motility-metrics.ipynb).
* Figure 5: [`3_motility-analysis-in-pools.ipynb`](notebooks/3_motility-analysis-in-pools.ipynb).
* Figure 6: [`4_compare-motility-in-pools-vs-wells.ipynb`](notebooks/4_compare-motility-in-pools-vs-wells.ipynb).
* Figure 7: [`5_pca-of-motility-features.ipynb`](notebooks/5_pca-of-motility-features.ipynb).

### Compute Specifications
Cell tracking was done on a Supermicro X12SPA-TF 64L running Ubuntu 22.04.1 with 512 GB RAM, 64 cores, and a 2 TB SSD.

The notebooks for statistical analysis were run on an Apple MacBook Pro with an Apple M3 Max chip running macOS Sonoma version 14.5 with 36 GB RAM, 14 cores, and 1TB SSD.


## Data
The full dataset underlying the pub is 355 GB and thus has been uploaded to the BioImage Archive (DOI: [10.6019/S-BIAD1298](https://doi.org/10.6019/S-BIAD1298)). To enable users to perform the analysis related to motility metrics, this repository provides CSV files containing summary motility statistics. More information is provided in [data/README.md](data/README.md).


## Scripts
There are four scripts located in [`src/chlamytracker/scripts`](src/chlamytracker/scripts), the first three of which are for processing biological image data, while the fourth was only run once to prepare the dataset for uploading to the BioImage Archive.
* `track_cells.py`: Track cells in raw brightfield time-lapse microscopy data.
* `make_movies_of_pools.py`: Render an animation of tracked cells in agar microchamber pools (after cell tracking).
* `make_movies_of_wells.py`: Render an animation of tracked cells in a microtiter plate (after cell tracking).
* `generate_bioimage_archive_file_lists.py`: Generate the lists of files needed for the BioImage Archive upload. (_No longer intended to be used._)

All scripts are configured with [`click`](https://click.palletsprojects.com/en/8.1.x/) such that
```bash
python src/chlamytracker/scripts/{script}.py --help
```
will display a help message that gives a description of what the script does as well as the arguments it accepts and their default values. The three scripts for processing biological image data also accept a `--glob` argument that can be used to filter the set of files to process. For example, to track cells from only one row of wells from a plate, one could run the command.

```bash
python src/chlamytracker/scripts/track_cells_in_wells.py \
    /path/to/directory/of/nd2/files/ \
    --glob "WellB*.nd2"
```

For more information on glob patterns, check out the official Python [documentation](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob) for the pathlib library. The default glob pattern is `"*.nd2"`.

### Cell tracking
`track_cells.py` executes cell tracking on a batch of time-lapse microscopy data. For accurate cell tracking, the script will first segment cells within each time lapse. The segmentation algorithm is effectively just background subtraction and intensity thresholding; see the section on "Tracking cells and motility phenotyping" of the [pub](https://doi.org/10.57844/arcadia-2d61-fb05) for details. Cell tracking is done using [btrack](https://btrack.readthedocs.io/en/latest/index.html). The output for each ND2 file is a TIFF file of the segmented timelapse and a CSV file of the motility data that contains every cell detected in the segmentation.

Microscopy data for the pub is comprised of cells swimming inside one of two different types of "vessels": either agar microchamber pools or microtiter plates. By default, the script expects cells to be swimming in a microtiter plate, but the `--vessel` argument can be used to change the expected vessel type as shown in the examples below. Regardless of the vessel type, the expected input is more or less the same: a ~20 sec timelapse of brightfield microscopy data stored as a ND2 file in which there are clearly unicellular organisms swimming around. There are no constraints on the duration, dimensions, frame rate, or pixel size of the timelapse, but the code has thus far predominantly been tested on 20 sec timelapses with dimensions around (400, 1200, 1200) T, Y, X acquired at 20–30 frames per second. Most cell tracking has been performed on different species and strains of _Chlamydomonas_, hence the default of 6 µm for the `min_cell_diameter_um` parameter. This parameter should be increased or decreased based on the size of the organism recorded.

To track cells in time-lapse videos of 384- or 1536-well plates, parallelized by [`dask`](https://image.dask.org/en/latest/):
```bash
python src/chlamytracker/scripts/track_cells.py \
    /path/to/directory/of/nd2/files/ \
    --output-directory /path/to/writeable/storage/location/ \
    --use-dask
```

To track cells in time-lapse data of 100 µm diameter agar microchamber pools, using 6 cores in parallel:
```bash
python src/chlamytracker/scripts/track_cells.py \
    /path/to/directory/of/nd2/files/ \
    --output-directory /path/to/writeable/storage/location/ \
    --pool-radius 50 \
    --num-cores 6
```

Note that in the above examples, `--output-directory` is an optional argument. If not provided, output will be written to a directory named `processed` within the input directory (first argument). If `{input-directory}/processed/` already exists, files may be overwritten.

### Making movies of tracked cells
To provide some sort of visual confirmation that the segmentation and cell tracking was done successfully, there are also scripts for adding animations of cell trajectories to the tracked cells using the napari plugin [napari-animation](https://github.com/napari/napari-animation). Here the choice for which script to run depends on the type of vessel used in the experiment.

To create animations of tracked cells in 384- or 1536-well plates at 20 fps.
```bash
python src/chlamytracker/scripts/make_movies_of_wells.py \
    /path/to/directory/of/nd2/files/ \
    --framerate 20
    --output-directory /path/to/writeable/storage/location/
```

To create animations of tracked cells in agar microchamber pools at 30 fps.
```bash
python src/chlamytracker/scripts/make_movies_of_pools.py \
    /path/to/directory/of/nd2/files/ \
    --framerate 30
    --output-directory /path/to/writeable/storage/location/
```

The output for each ND2 file is a MP4 file that is a compressed, contrast-enhanced version of the timelapse with cell trajectories animated in a variety of colors corresponding to the trajectory ID. Note that in the above examples, `--framerate` and `--output-directory` are both optional arguments. The default frame rate is 30 fps, while the default output directory is a directory named `processed` within the input directory (first argument). If `{input-directory}/processed/` already exists, files may be overwritten.


## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
