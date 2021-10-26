import os
import multiprocessing as mp
import pandas as pd


def run_batchfile(each_file):
    os.system(each_file)
    return 1


def get_each_df(each_dir):
    each_df = pd.read_csv(os.path.join(each_dir, 'image.csv'))
    each_df['group'] = os.path.basename(each_dir)
    return each_df


if __name__ == '__main__':
    para_text = "{\"row0\":644, \"row1\":924, \"col0\":840, \"col1\":1380, \"median_blur_size\":7, \"sobel_size\":5, \"dark_frame\":1, \"gblur_size\":10, \"crop_right_extra\":0, \"row_offset\":40, \"col_offset\":30, \"border\":[300], \"color_channel\":[\"gray\"], \"signal_type\":[\"spot\"], \"do_detrend\":1, \"ndetrend\":5, \"nsignal\":3, \"nblur\":400, \"top_fraction\":0.05, \"contour_mode\":1, \"contour_method\":2, \"rect_height\":10, \"edge_gap\":40}"

    # find sub directories
    dir = [each[0] for each in os.walk('.')]
    dir = dir[1:]  # eliminate the first one, which is '.'
    dir = [each for each in dir if ('.idea' not in each and '__pycache__' not in each)]

    # make batch files
    file = [os.path.basename(each) + '.bat' for each in dir]
    for each_dir, each_file in zip(dir, file):
        out_string = 'python "Hamilton_image_analysis.py" ' + '"' + each_dir + '" "' + para_text.replace('"',
                                                                                                         '\\"') + '"\n'
        out_file = open(each_file, 'w')
        out_file.write(out_string)
        out_file.close()

    # run batch files
    pool = mp.Pool(mp.cpu_count())
    pool.map(run_batchfile, file)
    pool.close()
    pool.join()

    # now compile all data
    df = pd.concat([get_each_df(each_dir) for each_dir in dir])
    df['guid'] = df['filename'].str.split('_', n=1, expand=True).iloc[:,0].astype(int)
    df.to_csv('all_results.csv', index=False)

