# 2024-unicellular-tracking

[![run with conda](https://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/projects/miniconda/en/latest/)
[![Arcadia Pub](https://img.shields.io/badge/Arcadia-Pub-596F74.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDI3LjcuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCA0My4yIDQwLjQiIHN0eWxlPSJlbmFibGUtYmFja2dyb3VuZDpuZXcgMCAwIDQzLjIgNDAuNDsiIHhtbDpzcGFjZT0icHJlc2VydmUiPgo8c3R5bGUgdHlwZT0idGV4dC9jc3MiPgoJLnN0MHtmaWxsOm5vbmU7c3Ryb2tlOiNGRkZGRkY7c3Ryb2tlLXdpZHRoOjI7c3Ryb2tlLWxpbmVqb2luOmJldmVsO3N0cm9rZS1taXRlcmxpbWl0OjEwO30KPC9zdHlsZT4KPGc+Cgk8cG9seWdvbiBjbGFzcz0ic3QwIiBwb2ludHM9IjIxLjYsMyAxLjcsMzcuNCA0MS41LDM3LjQgCSIvPgoJPGxpbmUgY2xhc3M9InN0MCIgeDE9IjIxLjYiIHkxPSIzIiB4Mj0iMjEuNiIgeTI9IjI3LjMiLz4KCTxwb2x5bGluZSBjbGFzcz0ic3QwIiBwb2ludHM9IjEyLjIsMTkuNCAyNC42LDMwLjEgMjQuNiwzNy40IAkiLz4KCTxsaW5lIGNsYXNzPSJzdDAiIHgxPSIxNy42IiB5MT0iMTYuNyIgeDI9IjE3LjYiIHkyPSIyNC4xIi8+Cgk8bGluZSBjbGFzcz0ic3QwIiB4MT0iMjguNiIgeTE9IjE1LjIiIHgyPSIyMS43IiB5Mj0iMjIuMSIvPgoJPHBvbHlsaW5lIGNsYXNzPSJzdDAiIHBvaW50cz0iNi44LDI4LjcgMTkuNSwzNC40IDE5LjUsMzcuNCAJIi8+Cgk8bGluZSBjbGFzcz0ic3QwIiB4MT0iMzQuOCIgeTE9IjI1LjgiIHgyPSIyNC42IiB5Mj0iMzYuMSIvPgoJPGxpbmUgY2xhc3M9InN0MCIgeDE9IjI5LjciIHkxPSIyMi4yIiB4Mj0iMjkuNyIgeTI9IjMwLjkiLz4KPC9nPgo8L3N2Zz4K)](https://doi.org/10.57844/arcadia-2d61-fb05)


## Purpose

Detect and track unicellular organisms for scaling up high-throughput motility assay development.

![movie](../resources/cell-tracks-in-pools.gif)


## Installation and setup

This repository uses conda to manage software environments and installations. If you do not already have conda installed, you can find operating system-specific instructions for installing miniconda [here](https://docs.anaconda.com/miniconda/). After installing conda, navigate to a directory where you would like to clone the repository to, and run the following commands to create the pipeline run environment.

```{bash}
git clone https://github.com/Arcadia-Science/2024-unicellular-tracking.git
conda env create -n unicellular-tracking --file envs/dev.yml
conda activate unicellular-tracking
pip install -e .
```

If the installation was successful, the below command will return without error.
```{bash}
python -c "import chlamytracker"
```

## Overview

### Description of the folder structure

This repository is organized into the following top-level directories.
* **btrack_config**: YAML file for [`btrack`](https://btrack.readthedocs.io/en/latest/index.html) configuration.
* **data**: CSV files containing summary motility metrics from measured cell trajectories.
* **envs**: YAML file including the packages and dependencies used for creating the conda environment.
* **notebooks**: 
* **resources**: 
* **results**: A collection of output files used for the analysis described in the pub as well as for generating the figures.
* **src/chlamytracker**: Source code



### Methods

TODO: Include a brief, step-wise overview of analyses performed.

> Example:
>
> 1.  Download scripts using `download.ipynb`.
> 2.  Preprocess using `./preprocessing.sh -a data/`
> 3.  Run Snakemake pipeline `snakemake --snakefile Snakefile`
> 4.  Generate figures using `pub/make_figures.ipynb`.

### Compute Specifications

TODO: Describe what compute resources were used to run the analysis. For example, you could list the operating system, number of cores, RAM, and storage space.

## Data
The full dataset underlying the pub is in excess of 300 GB and thus has been uploaded to the BioImage Archive (DOI: [10.6019/S-BIAD1298](https://doi.org/10.6019/S-BIAD1298)). To enable users to perform the analysis related to motility metrics, this repository provides CSV files containing summary motility statistics. More information is provided in [data/](data/README.md).

## Scripts

Scripts are located in `2024-unicellular-tracking/src/chlamytracker/scripts/`. Examples below assume this is your current working directory. 

All scripts are configured with [click](https://click.palletsprojects.com/en/8.1.x/) such that

```bash
python {script}.py --help
```

will display a help message that gives a description of what the script does as well as the arguments it accepts and their default values. All scripts accept a `--glob` argument that can be used to filter the set of files to process. For example, to track cells from only one row of wells from a plate, one could run the command.

```bash
python track_cells_in_wells.py /path/to/directory/of/nd2/files/ --glob "WellB*.nd2"
```

For more information on glob patterns check out the official Python [documentation](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob) for the pathlib library. The default glob pattern is "*.nd2".

### Cell tracking
There are two scripts for executing cell tracking on a batch of time-lapse microscopy data. Which script to run depends on the type of vessel used in the experiment: either agar microchamber pools or 384 (or 1536) well plates. Regardless of the vessel type, however, the expected input is more or less the same: a ~20 sec timelapse of brightfield microscopy data stored as a nd2 file in which there are clearly unicellular organisms swimming around. There are no constraints on the duration, dimensions, frame rate, or pixel size of the timelapse, but the code has thus far predominantly been tested on 20 sec timelapses with dimensions around (400, 1200, 1200) T, Y, X acquired at 20-30 frames per second. Most cell tracking has been performed on different species and strains of Chlamydomonas, hence the default of 6 µm for the `min_cell_diameter_um` parameter. This parameter should be increased or decreased based on the size of the organism recorded.

Both scripts will perform segmentation and cell tracking on the timelapse. The segmentation is effectively just background subtraction and intensity thresholding; cell tracking is done using [btrack](https://btrack.readthedocs.io/en/latest/index.html). The output for each nd2 file is a tiff file of the segmented timelapse and a csv file of the motility data that contains every cell detected in the segmentation.

To track cells in timelapse videos of 384 or 1536 well plates.
```bash
python track_cells_in_wells.py /path/to/directory/of/nd2/files/ --output-directory /path/to/writeable/storage/location/ --use-dask
```

To track cells in timelapse videos of 100µm diameter agar microchamber pools.
```bash
python track_cells_in_pools.py /path/to/directory/of/nd2/files/ --output-directory /path/to/writeable/storage/location/ --pool-radius 50
```

Notes
* These scripts will attempt to output to `{input-directory}/processed/` if  the `--output-directory` option is not provided.
* If `{input-directory}/processed/` already exists, files may be overwritten.


### Making movies of tracked cells
To provide some sort of visual confirmation that the segmentation and cell tracking was done successfully, there are also scripts for adding animations of cell trajectories to the tracked cells using the napari plugin [napari-animation](https://github.com/napari/napari-animation). Here too the choice for which script to run depends on the type of vessel used in the experiment.

To create animations of tracked cells in 384 or 1536 well plates.
```bash
python make_movies_of_wells.py /path/to/directory/of/nd2/files/ --output-directory /path/to/writeable/storage/location/
```

To create animations of tracked cells in agar microchamber pools.
```bash
python make_movies_of_pools.py /path/to/directory/of/nd2/files/ --output-directory /path/to/writeable/storage/location/
```

The output for each nd2 file is a mp4 file that is basically a compressed, contrast-enhanced version of the timelapse with cell trajectories animated in a variety of colors corresponding to the trajectory ID.


## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
