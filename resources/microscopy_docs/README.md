# Microscopy resources

## Contents
This folder contains HTML and BIN files corresponding to the automated microscopy workflows created through the Nikon NIS-Elements JOBS software. As discussed in the [pub](https://doi.org/10.57844/arcadia-2d61-fb05), automated time-lapse microscopy was done using two separate microscopes. A similar but distinct acquisition workflow was thus developed for each microscope.

* `Timelapse-Grid-10x-100um-pools.bin`: Nikon JOBS workflow for imaging microchamber pools used on an upright Nikon Ni-E microscope.
* `Timelapse-Grid-10x-100um-pools.html`: HTML preview of `Timelapse-Grid-10x-100um-pools.bin`
* `Timelapse-Grid-10x-384-well-plate.bin`: Nikon JOBS workflow for imaging 384-well plates used on an inverted Nikon Ti2-E microscope.
* `Timelapse-Grid-10x-384-well-plate.html`: HTML preview of `Timelapse-Grid-10x-384-well-plate.bin`

The Nikon JOBS workflows (BIN files) can be imported via the JOBS Manager window in the NIS-Elements software.

## Automated acquisition workflow
We implemented the following automation procedure using the Nikon JOBS automation software:

* **CaptureDefinition**: Defined the optical configuration for the 10× objective, the Kinetix camera, and closing the active shutter between imaging.
* **StageArea**: Mapped the xyz position of the four corners of our square piece of agar microchambers. We used a polygon setting for the shape because the square wasn’t always perfectly aligned on the slide. 
* **FocusSurface**: Created a map of the focus surface based on the positions defined above, so that the focal plane accounted for slight differences in the height of the agar. We checked the box to compute z-values from a “smooth interpolation surface.”
* **GeneratedPoints**: Defined the working area and the number of fields of view (FOV) that we wanted to acquire based on the stage area. Here, we selected that we wanted the FOV to cover the defined area and set the scan direction to “meander.” We set the number of FOV depending on the size of the piece of agar microchamber and the number of microchambers that contained 1–3 cells. If cells were sparse due to differences in sample loading, then we acquired additional FOVs. 
* **Point Loop**: For each of the points in generated points (each FOV), we tell the microscope to a) FocusToSurface (using step 3), b) Acquire FastTimelapse (defined as 400 frames with the capture definition from step 1), and c) Storage (where to store each FOV).

The workflow for the 384-well plate on the inverted microscope was identical, except we no longer needed to define the stage area or do the focus surface step. Instead, we defined the type of plate used (Cellvis 384-well), selected the wells we wanted to image, and set the position to move to the center of the well before acquiring a single FOV per well.

While we used Nikon JOBS, other software should work just as well to set up automations in a similar fashion.
