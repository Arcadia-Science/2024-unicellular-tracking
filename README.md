# 2024-unicellular-tracking

[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/projects/miniconda/en/latest/)

## Purpose

Detect and track unicellular organisms for scaling up high-throughput motility assay development.

## Installation and Setup

This repository uses conda to manage software environments and installations.

```{bash}
conda env create -n tracking --file envs/dev.yml
conda activate tracking
```

## Overview

### Description of the folder structure

TODO: Write description.

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

## Scripts

Scripts are located in `2024-unicellular-tracking/src/chlamytracker/scripts/`. Examples below assume this is your current working directory. All scripts are configured with [click](https://click.palletsprojects.com/en/8.1.x/) such that

```bash
python {script}.py --help
```

will display a help message that gives a description of what the script does as well as the arguments it accepts and their default values. All scripts accept a `--glob` argument that can be used to filter the set of files to process. For example, to track cells from only one row of wells from a plate, one could run the command.

```bash
python track_cells_in_wells.py /path/to/directory/of/nd2/files/ --glob "WellB*.nd2"
```

For more information on glob patterns check out the official Python [documentation](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob) for the pathlib library. The default glob pattern is "*.nd2".

### Cell tracking
There are two scripts for executing cell tracking on a batch of timelapse microscopy data. Which script to run depends on the substrate used in the experiment: either agar microchamber pools or 384 (or 1536) well plates. Regardless of the substrate, however, the expected input is more or less the same: a ~20 sec timelapse of brightfield microscopy data stored as a nd2 file in which there are clearly unicellular organisms swimming around. There are no constraints on the duration, dimensions, frame rate, or pixel size of the timelapse, but the code has thus far predominantly been tested on 20 sec timelapses with dimensions around (400, 1200, 1200) T, Y, X acquired at 20-50 frames per second. Most cell tracking has been performed on different species and strains of Chlamydomonas, hence the default of 6 µm for the `min_cell_diameter_um` parameter. This parameter should be increased or decreased based on the size of the organism recorded.

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
To provide some sort of visual confirmation that the segmentation and cell tracking was done successfully, there are also scripts for adding animations of cell trajectories to the tracked cells using the napari plugin [napari-animation](https://github.com/napari/napari-animation). Here too the choice for which script to run depends on the substrate used in the experiment.

To create animations of tracked cells in 384 or 1536 well plates.
```bash
python make_movies_of_wells.py /path/to/directory/of/nd2/files/ --output-directory /path/to/writeable/storage/location/
```

To create animations of tracked cells in agar microchamber pools.
```bash
python make_movies_of_pools.py /path/to/directory/of/nd2/files/ --output-directory /path/to/writeable/storage/location/
```

The output for each nd2 file is a mp4 file that is basically a compressed, contrast-enhanced version of the timelapse with cell trajectories animated in a variety of colors corresponding to the trajectory ID.

## Data

TODO: Add videos that show the expected input/output.

TODO: Add details about the description of input / output data and links to Zenodo depositions, if applicable.


## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
