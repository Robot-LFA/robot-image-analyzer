# Image analysis on the Hamilton robot

## Overview

This software can analyze spots or lines, with dark frames (for strips in IVL's custom holder with dark background) or light frames (for strips in cassettes made of white plastic).

It can analyze full images, or images manually cropped to the regions of interests (ROIs).

It can also analyze multiple signals on the same strips (typically a test line and a control line). There can be any positive number of signals, but the more the harder it is to automatically choose the ROIs due to variations in strips.

## Usage

### General information
Please see example folders.

Currently, please make a copy of [backend.py](backend.py) and either [Hamilton_image_analysis.py](Hamilton_image_analysis.py) or [manual_ROI.py](manual_ROI.py) to use for each analysis. This is not the best practice but it does allow one to keep track of what has been done.

Use files [troubleshoot.py](troubleshoot.py) and [troubleshoot_folder.py](troubleshoot_folder.py) to troubleshoot and tinker with parameters. Usually changes in initial ROIs, offsets, and borders are effective in getting correct results.

Investigate output files in the folder containing the images to see if image analysis is done correctly. Example: [example_line_darkframe/factorial_experiment_01_worklist/image_ROI_line.pdf](example_line_darkframe/factorial_experiment_01_worklist/image_ROI_line.pdf)

Numerical results are in the csv files. Example: [example_line_darkframe/factorial_experiment_01_worklist/image.csv](example_line_darkframe/factorial_experiment_01_worklist/image.csv)

Sometimes it is faster to choose the ROI manually. For example, there may be only a few images in total, or there are only a few images with problems after tinkering with parameters. Example: [example_line_manualROI](example_line_manualROI).

ImageJ is a great tool to find initial ROIs, manually choose final ROIs, and choose color channels (e.g. to avoid blood background). 



### Options to use the software (explained via examples):
* Use python to run [example_line_lightframe/generate_batch_files_and_run.py](example_line_lightframe/generate_batch_files_and_run.py) to generate batch files for different folders, run those batch files, and put together results. The batch files can be manually run later as executables. Example batch file: [example_line_lightframe/factorial_experiment0_worklist.bat](example_line_lightframe/factorial_experiment0_worklist.bat)
* Manually make batch files to run on the command line or by double-clicking. These batch files are the same as those generated automatically above for the same analysis. Example batch file to analyze manually chosen ROI: [example_line_manualROI/manual_ROI.bat](example_line_manualROI/manual_ROI.bat)

### Parameters

The user has to specify parameter text in either the python code to generate batch files or in the batch files.

Example:
```
"{\"row0\":608, \"row1\":900, \"col0\":680, \"col1\":1500, \"median_blur_size\":7, \"sobel_size\":5, \"dark_frame\":0, \"gblur_size\":30, \"crop_right_extra\":1, \"row_offset\":40, \"col_offset\":30, \"border\":[190, 465], \"color_channel\":[\"red\", \"red\"], \"signal_type\":[\"line\", \"line\"], \"do_detrend\":1, \"ndetrend\":5, \"nsignal\":3, \"nblur\":400, \"top_fraction\":0.05, \"contour_mode\":1, \"contour_method\":2, \"rect_height\":10, \"edge_gap\":40}"
```
The list of parameters is below. The ones in **bold** will usually need to be modified for a new strip configuration.

#### To find initial, rough ROI:
* **row0**: initial top row
* **row1**: initial bottom row
* **col0**: initial left column
* **col1**: initial right column

#### To crop top and bottom:
* median_blur_size: size to median blur, before getting labeled regions to get slopes
* sobel_size: size of kernel for blurring during sobel calculation, to crop top and bottom
* **dark_frame**: if True: the area around the strip is black, if False, white

#### To crop left and right:
* gblur_size: size of kernel for gaussian blurring
* crop_right_extra: if True, on the right, choose the last valley before the last peak, if False, choose the last valley

#### To split into ROIs for different signals (e.g. test line and control line)
* row_offset: row offset to avoid boundary effects
* col_offset: column offset to avoid boundary effects
* **border**: border to split test and control images, a list if there are more than 2 test lines or spots

#### To process signals (line and spot)
* **color_channel**: choices: 'red', 'green', 'blue', and anything else means 'gray'
* **signal_type**: line or spot, in a list there are multiple ones, such test and control lines
* do_detrend: detrend or not
* ndetrend: number of points at the beginning and the end of the series to used for detrending
* nsignal: number of points at the max and min ranges to average over to take the delta to find the signal

#### To process signals (spot only)
* nblur: number of times for gaussian blurring with kernel size of 5
* top_fraction: top fraction for threshold finding
* contour_mode: mode to find contours (see open cv documentation)
* contour_method: method to find contours (see open cv documentation)
* rect_height: rectangle height from the contour center to find the ROI
* edge_gap: if the contour center is within the edge_gap from the top or bottom borders, use the middle

### Usual workflow

* Use ImageJ to find initial ROIs.
* Run image analysis software through images.
* Investigate results.
* Tinker with parameters.
* Repeat until most images are properly analyzed.
* Manually make and analyze ROIs if there are a few outliers. Keep parameters to process signals the same. 

