from backend import *
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_folder', help='input folder containing images')
parser.add_argument('parameters', help='parameters for image analysis')
args = parser.parse_args()

print(args)

para = json.loads(args.parameters)


def method(image):
    return process_ROI(image, **para)


process_folder_manual_ROI(args.input_folder, method, 'image')

# para_text = "{\"color_channel\":\"red\", \"signal_type\":\"line\", \"do_detrend\":1, \"ndetrend\":5, \"nsignal\":3, \"sobel_size\":5, \"nblur\":400, \"top_fraction\":0.05, \"contour_mode\":1, \"contour_method\":2, \"rect_height\":10, \"edge_gap\":40}"
#
# para = json.loads(para_text)
#
# def method(image):
#     return process_ROI(image, **para)
#
#
# process_folder_manual_ROI("manual_ROI", method, 'image')

