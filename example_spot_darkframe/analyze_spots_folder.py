from backend import *
import json
import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument('input_folder', help='input folder containing images')
# parser.add_argument('parameters', help='parameters for image analysis')
# args = parser.parse_args()
#
# print(args)
#
# para = json.loads(args.parameters)

para_text = "{\"row0\":644, \"row1\":924, \"col0\":840, \"col1\":1380, \"median_blur_size\":7, \"sobel_size\":5, \"dark_frame\":1, \"gblur_size\":10, \"crop_right_extra\":0, \"row_offset\":40, \"col_offset\":30, \"border\":[300], \"color_channel\":[\"gray\"], \"signal_type\":[\"spot\"], \"do_detrend\":1, \"ndetrend\":5, \"nsignal\":3, \"nblur\":400, \"top_fraction\":0.05, \"contour_mode\":1, \"contour_method\":2, \"rect_height\":10, \"edge_gap\":40}"


para = json.loads(para_text)

def method(image):
    return process_image(image, **para)


# process_folder(args.input_folder, process_image, 'image')

process_folder('spots/2_imaging_worklist', method, 'image')
