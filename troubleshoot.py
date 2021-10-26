from backend import *
import json
import cv2 as cv


input_file = 'example_spot_darkframe/2_imaging_worklist/9_20190730_122541.png'

# set up parameters
para_text = "{\"row0\":644, \"row1\":924, \"col0\":840, \"col1\":1380, \"median_blur_size\":7, \"sobel_size\":5, \"dark_frame\":1, \"gblur_size\":20, \"row_offset\":20, \"col_offset\":20, \"border\":[300], \"color_channel\":[\"gray\"], \"signal_type\":[\"spot\"], \"do_detrend\":1, \"ndetrend\":5, \"nsignal\":3, \"nblur\":400, \"top_fraction\":0.05, \"contour_mode\":1, \"contour_method\":2, \"rect_height\":10, \"edge_gap\":40}"

para = json.loads(para_text)

# read image
image = cv.imread(input_file)
result = process_image(image, **para)

print(result)
