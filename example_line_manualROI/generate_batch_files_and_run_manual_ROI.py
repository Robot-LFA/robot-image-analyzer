import os


def run_batchfile(each_file):
    os.system(each_file)
    return 1


if __name__ == '__main__':
    para_text = "{\"color_channel\":\"red\", \"signal_type\":\"line\", \"do_detrend\":1, \"ndetrend\":5, \"nsignal\":3, \"sobel_size\":5, \"nblur\":400, \"top_fraction\":0.05, \"contour_mode\":1, \"contour_method\":2, \"rect_height\":10, \"edge_gap\":40}"

    # find sub directories
    dir = 'manual_ROI'

    # make batch files
    file = dir + '.bat'
    out_string = 'python "manual_ROI.py" ' + '"' + dir + '" "' + para_text.replace('"', '\\"') + '"\n'
    out_file = open(file, 'w')
    out_file.write(out_string)
    out_file.close()

    run_batchfile(file)
