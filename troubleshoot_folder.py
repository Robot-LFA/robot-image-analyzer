from backend import *
import json

para_text = "{\"row0\":608, \"row1\":900, \"col0\":680, \"col1\":1500, \"median_blur_size\":7, \"sobel_size\":5, \"dark_frame\":0, \"gblur_size\":30, \"crop_right_extra\": 1, \"row_offset\":40, \"col_offset\":20, \"border\":[190, 470], \"color_channel\":[\"red\", \"red\"], \"signal_type\":[\"line\", \"line\"], \"do_detrend\":1, \"ndetrend\":5, \"nsignal\":3, \"nblur\":400, \"top_fraction\":0.05, \"contour_mode\":1, \"contour_method\":2, \"rect_height\":10, \"edge_gap\":40}"

para = json.loads(para_text)


def method(image):
    return process_image(image, **para)


process_folder('example_line_lightframe/factorial_experiment0_worklist', method, 'image')
