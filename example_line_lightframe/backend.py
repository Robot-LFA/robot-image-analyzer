import numpy as np
import cv2 as cv
from scipy.ndimage.measurements import label
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def norm(image, scale_to=255):
    '''
    :param image: input image
    :param scale_to: max to scale to, default: 255 for 8 bit images
    :return:
    '''
    out = image[:].astype(np.float)
    out = out - out.min()
    out = out / out.max() * scale_to
    out = np.uint8(out.astype(np.int))
    return out


def get_rotated_image(img, theta):
    '''
    :param img: input image
    :param theta: angle to rotate
    :return: rotated image
    '''
    # angle: in rad
    rows, cols = img.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), theta * 180 / np.pi, 1)
    return cv.warpAffine(img, rotation_matrix, (cols, rows))


def find_valley_peak(line):
    """
    :param line: list, series of values in a line
    :return: indices of valleys and peaks
    """
    line = np.array(line)
    delta1 = line[1:-1] - line[:-2]
    delta2 = line[2:] - line[1:-1]
    ivalley = np.where((delta1 < 0) * (delta2 > 0))[0] + 1
    ipeak = np.where((delta1 > 0) * (delta2 < 0))[0] + 1
    return ivalley, ipeak


def crop_left_right_cassette(image, gblur_size=30, crop_right_extra=True):
    """
    crop left and right
    :param image: input image
    :param gblur_size: size of kernel for gaussian blurring
    :param crop_right_extra: if True, on the right, choose the last valley before the last peak, if False, choose the last valley
    :return: cropped image
    """
    if len(image.shape) == 2:
        gray = np.copy(image)
    elif len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif len(image.shape) > 3:
        gray = image.mean(axis=-1)

    line = gray.mean(axis=0)

    line_gb = gaussian_filter1d(line, gblur_size, order=1)
    ivalley, ipeak = find_valley_peak(line_gb)
    col0 = ipeak[ipeak > ivalley[0]][0]
    if crop_right_extra:
        col1 = ivalley[ivalley < ipeak[-1]][-1]
    else:
        col1 = ivalley[-1]

    return {'image': gray[:, col0:col1],
            'crop_range': np.array([col0, col1])}


def simple_crop(image, row0=630, row1=950, col0=950, col1=1600):
    """
    :param image: input image
    :param row0: top row
    :param row1: bottom row
    :param col0: left column
    :param col1: right column
    :return: cropped image
    """
    return image[row0:row1, col0:col1]


def get_slope(x, y):
    """
    get slope from x and y series, using numpy.polyfit. return 0 if there is an exception
    :param x: numpy array, x series
    :param y: numpy array, y series
    :return: slope
    """
    try:
        slope = np.polyfit(x, y, deg=1)[0]
    except:
        slope = 0
    return(slope)


def align_crop(image, median_blur_size=7, sobel_size=5, dark_frame=1, sign_list=[1, -1]):
    '''
    to align and crop top bottom
    :param image: input image
    :param row0: initial top row
    :param row1: initial bottom row
    :param col0: initial left column
    :param col1: initial right column
    :param median_blur_size: size to median blur, before getting labeled regions to get slopes
    :param sobel_size: size of kernel for blurring during sobel calculation, to crop top and bottom
    :param dark_frame: if True: the area around the strip is black, if False, white
    :param sign_list: to keep track of sign of the derivative
    :return: ROI, angle used for rotation, and status (with/without strip)
    '''
    # convert to grayscale anyway just in case
    if len(image.shape) == 2:
        gray = np.copy(image)
    elif len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif len(image.shape) > 3:
        gray = image.mean(axis=-1)

    # find horizontal lines, then find the angle
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_size)
    threshold_value, threshold_image = cv.threshold(norm(np.abs(sobely)), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    threshold_image = cv.medianBlur(threshold_image, median_blur_size)
    labeled_array, num_features = label(threshold_image)
    # get the slope and the number of pixels
    slope = np.array(
        [[get_slope(*np.where(labeled_array == i)[::-1]), len(np.where(labeled_array == i)[0])]
         for i in range(1, num_features + 1)])
    slope = slope[~np.isnan(slope[:, 0]),]
    angle = np.arctan(slope[:, 0])
    # take the mean weighted by the number of pixels
    angle_mean = (angle * slope[:, 1]).sum() / slope[:, 1].sum()

    # rotate gray and sobely
    gray_rot = get_rotated_image(gray, angle_mean)
    sobely_rot = get_rotated_image(sobely, angle_mean)

    # find the top and bottom borders
    line = sobely_rot.sum(axis=1)
    mid = int(len(line) / 2)

    if dark_frame == 1:
        sign_list = np.array(sign_list)
    else:
        sign_list = np.array(sign_list) * (-1)
    i0 = np.argmax(sign_list[0] * line[:mid])
    i1 = np.argmax(sign_list[1] * line[mid:]) + mid

    # also return the angle and row0, row1 to use for cropping original images
    return {'image': gray_rot[i0:i1],
            'angle': angle_mean,
            'crop_range': np.array([i0, i1])}


def get_sub_ROI(image, row_offset=5, col_offset=5, border=340):
    """
    get sub ROIs from the initial ROIs using user specified borders
    :param image: initial ROI
    :param row_offset: row offset to avoid boundary effects
    :param col_offset: column offset to avoid boundary effects
    :param border: border to split test and control images, a list if there are more than 2 test lines or spots
    :return: dictionary, for test and control images
    """
    border = np.array([border]).flatten()  # in case border is a single number, convert to array
    if border[-1] > image.shape[1]-col_offset:
        border[-1] = image.shape[1]-col_offset
    start = np.append([col_offset], border[:-1])

    sub = [image[row_offset:-row_offset, col0:col1] for col0, col1 in zip(start, border)]

    image_out = image.copy()
    for col0, col1 in zip(start, border):
        cv.rectangle(image_out, (col0, row_offset), (col1, image.shape[0]-row_offset),
                     color=(0, 255, 0), thickness=3, lineType=1)
    return {'sub_list': sub,
            'image': image_out}


def detrend_line_linear(line, ndetrend=3):
    """
    detrend line, using beginning and end values to do a linear correction
    :param line: numpy array, values along the line
    :param ndetrend: number of points at the beginning and the end to use to detrend
    :return: detrended line
    """
    x = np.arange(line.shape[0])
    x = np.append(x[:ndetrend], x[-ndetrend:])
    y = line[x]
    fit = np.polyfit(x, y, 1)
    slope, intercept = fit
    to_subtract = slope * np.arange(line.shape[0]) + intercept
    return (line - to_subtract)


def pick_channel(image, color_channel):
    """
    pick the specified channel
    :param image: input image
    :param color_channel: choices: 'red', 'green', 'blue', and anything else means 'gray'
    :return:
    """
    color_channel = color_channel.lower()
    if color_channel == 'red':
        out = image[:, :, 2]
    elif color_channel == 'green':
        out = image[:, :, 1]
    elif color_channel == 'blue':
        out = image[:, :, 0]
    else:
        out = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return out


def get_line_signal(ROI, do_detrend, ndetrend=5, nsignal=3):
    """
    get line signal
    :param ROI: input image
    :param do_detrend: if True, do detrend, if False, not
    :param ndetrend: number of points at the beginning and the end to used for detrending
    :param nsignal: number of points at the max and min ranges to average over to take the delta to find the signal
    :return: dictionary, line, signal, ROI
    """
    line = 255 - ROI.mean(axis=0)
    if do_detrend == 1:
        line = detrend_line_linear(line, ndetrend)
    line_sorted = np.copy(line)
    line_sorted.sort()
    signal = (line_sorted[-nsignal:] - line_sorted[:nsignal]).mean()
    return {'line': line,
            'signal': signal,
            'image': ROI}


def get_spot_signal(ROI,
                    sobel_size, nblur, top_fraction, contour_mode, contour_method, rect_height, edge_gap,
                    do_detrend, ndetrend, nsignal):
    """
    get spot signals
    :param ROI: input image
    :param sobel_size: size of kernel to do sobel in the x direction to find the contours to detect spots
    :param nblur: number of times for gaussian blurring with kernel size of 5
    :param top_fraction: top fraction for threshold finding
    :param contour_mode: mode to find contours (see open cv documentation)
    :param contour_method: method to find contours (see open cv documentation)
    :param rect_height: rectangle height from the contour center to find the ROI
    :param edge_gap: if the contour center is within the edge_gap from the top or bottom borders, use the middle
    :param do_detrend: detrend or not
    :param ndetrend: number of points at the beginning and the end of the series to used for detrending
    :param nsignal: number of points at the max and min ranges to average over to take the delta to find the signal
    :return: dictionary, line, signal, original ROI, horizontal lines specifying the top and bottom of the ROI used
    """
    ROI_center_contours = center_contours(ROI,
                                          sobel_size=sobel_size,
                                          nblur=nblur,
                                          top_fraction=top_fraction,
                                          contour_mode=contour_mode,
                                          contour_method=contour_method)

    cX, cY = ROI_center_contours['center']
    contours = ROI_center_contours['contours']

    if cY <= edge_gap or cY >= ROI.shape[0] - edge_gap or len(contours) == 0:
        cY, cX = (np.array(ROI.shape) / 2).round().astype(int)

    signal_row0 = max(0, int(round(cY - rect_height)))
    signal_row1 = min(ROI.shape[0], int(round(cY + rect_height)))

    signalROI = ROI[signal_row0:signal_row1]

    line_signal = get_line_signal(signalROI, do_detrend, ndetrend, nsignal)

    image_out = ROI.copy()
    if len(contours) > 0:
        epsilon = 0.002 * cv.arcLength(contours, True)
        approx = cv.approxPolyDP(contours, epsilon, True)
        cv.drawContours(image_out, [approx], -1, (0, 255, 0), 1)
    cv.circle(image_out, center=(cX, cY), radius=2, color=(0, 255, 0), thickness=2, lineType=1)
    cv.rectangle(image_out, (0, signal_row0), (ROI.shape[1]-1, signal_row1),
                 color=(0, 255, 0), thickness=3, lineType=1)

    return {'line': line_signal['line'],
            'signal': line_signal['signal'],
            'image': image_out,
            'hline': [signal_row0, signal_row1]}


def process_ROI(ROI, color_channel, signal_type='line',
                do_detrend=1, ndetrend=5, nsignal=3,
                sobel_size=5, nblur=400,
                top_fraction=0.05, contour_mode=1, contour_method=2, rect_height=10, edge_gap=40):
    """
    process ROI
    :param ROI: input ROI
    :param color_channel: choices: 'red', 'green', 'blue', and anything else means 'gray'
    :param signal_type: spot or line
    :param do_detrend: detrend or not
    :param ndetrend: number of points at the beginning and the end of the series to used for detrending
    :param nsignal: number of points at the max and min ranges to average over to take the delta to find the signal
    :param sobel_size: for spot only, size of kernel for blurring during sobel calculation, to crop top and bottom
    :param nblur: for spot only, number of times for gaussian blurring with kernel size of 5
    :param top_fraction: for spot only, top fraction for threshold finding
    :param contour_mode: for spot only, mode to find contours (see open cv documentation)
    :param contour_method: for spot only, method to find contours (see open cv documentation)
    :param rect_height: for spot only, rectangle height from the contour center to find the ROI
    :param edge_gap: for spot only, if the contour center is within the edge_gap from the top or bottom borders, use the middle
    :return: dictionary, result, line, and image
    """
    out_image = pick_channel(ROI, color_channel)
    if signal_type == 'spot':
        out = get_spot_signal(out_image, sobel_size, nblur, top_fraction,
                              contour_mode, contour_method, rect_height, edge_gap,
                              do_detrend, ndetrend, nsignal)
    else:
        out = get_line_signal(out_image, do_detrend, ndetrend, nsignal)

    image_color = ROI.copy()
    if 'hline' in out.keys():
        for each in out['hline']:
            cv.line(image_color, (0, each), (image_color.shape[1]-1, each), (255, 0, 255), 2)
    out['image_color'] = image_color
    return out


def center_contours(ROI, sobel_size=5, nblur=200, top_fraction=0.05, contour_mode=1, contour_method=2):
    """
    calculate centers and contours to find spots
    :param ROI: input image
    :param sobel_size: size of kernel for blurring during sobel calculation in the x direction (flow direction)
    :param nblur: number of times for gaussian blurring with kernel size of 5
    :param top_fraction: top fraction for threshold finding
    :param contour_mode: mode to find contours (see open cv documentation)
    :param contour_method: method to find contours (see open cv documentation)
    :return: center and contour
    """
    # calculate x derivative
    sobelx = cv.Sobel(ROI, cv.CV_64F, 1, 0, ksize=sobel_size)
    sobelx = -sobelx
    sobelx[sobelx <= 0] = 0
    sobelx[sobelx > 0] = 255

    # blur agressively
    sobelx_gb = sobelx
    for i in range(nblur):
        sobelx_gb = cv.GaussianBlur(norm(sobelx_gb), (5, 5), 0)
    # find threshold at the top fraction
    sobelx_gb_flat = np.array(sobelx_gb).flatten()
    index = int(sobelx_gb_flat.shape[0] * top_fraction)
    sobelx_gb_flat.sort()
    sobelx_gb_flat = sobelx_gb_flat[::-1]
    threshold = sobelx_gb_flat[index]
    sobelx_gb[sobelx_gb <= threshold] = 0

    # find contours
    contours, hierarchy = cv.findContours(norm(sobelx_gb), mode=contour_mode, method=contour_method)

    # remove contours touching left edge
    contours = [each for each in contours if (each[:, :, 0] == 0).sum() == 0]
    # remove contours touching right edge
    contours = [each for each in contours if (each[:, :, 0] == ROI.shape[1]-1).sum() == 0]

    if len(contours) > 0:
        max_cnt = max(enumerate(contours), key=(lambda x: len(x[1])))[0]

        # find center of mass
        cnt = np.array(contours[max_cnt])[:, 0, :].transpose()
        cX, cY = np.average(cnt, axis=1, weights=ROI[tuple(cnt[::-1])]).round().astype(np.int)
        countours_out = contours[max_cnt]
    else:
        cY, cX = (np.array(ROI.shape) / 2).round().astype(int)
        countours_out = []

    return {'center': (cX, cY),
            'contours': countours_out}


def process_image(image,
                  row0, row1, col0, col1,  # for simple cropping
                  median_blur_size, sobel_size, dark_frame,  # for top bottom cropping
                  gblur_size, crop_right_extra,  # for left right cropping
                  row_offset, col_offset, border,  # for splitting into test and control ROIs
                  color_channel, signal_type,  # channel and type (line/spot)
                  do_detrend, ndetrend, nsignal,  # to process lines
                  nblur, top_fraction, contour_mode, contour_method, rect_height, edge_gap,  # to process spots
                  **kwargs):
    """
    process image
    :param image: input image
    :param row0: initial top row
    :param row1: initial bottom row
    :param col0: initial left column
    :param col1: initial right column
    :param median_blur_size: size to median blur, before getting labeled regions to get slopes
    :param sobel_size: size of kernel for blurring during sobel calculation, to crop top and bottom
    :param dark_frame: if True: the area around the strip is black, if False, white
    :param gblur_size: size of kernel for gaussian blurring
    :param crop_right_extra: if True, on the right, choose the last valley before the last peak, if False, choose the last valley
    :param row_offset: row offset to avoid boundary effects
    :param col_offset: column offset to avoid boundary effects
    :param border: border to split test and control images, a list if there are more than 2 test lines or spots
    :param color_channel: choices: 'red', 'green', 'blue', and anything else means 'gray'
    :param signal_type: line or spot, in a list there are multiple ones, such test and control lines
    :param do_detrend: detrend or not
    :param ndetrend: number of points at the beginning and the end of the series to used for detrending
    :param nsignal: number of points at the max and min ranges to average over to take the delta to find the signal
    :param nblur: number of times for gaussian blurring with kernel size of 5
    :param top_fraction: top fraction for threshold finding
    :param contour_mode: mode to find contours (see open cv documentation)
    :param contour_method: method to find contours (see open cv documentation)
    :param rect_height: rectangle height from the contour center to find the ROI
    :param edge_gap: if the contour center is within the edge_gap from the top or bottom borders, use the middle
    :param kwargs: in case there are unnecessary arguments
    :return: dictionary of result, image, and column offset for plotting
    """
    # gray
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    crop0 = simple_crop(gray, row0, row1, col0, col1)
    crop1 = align_crop(crop0, median_blur_size, sobel_size, dark_frame)
    crop2 = crop_left_right_cassette(crop1['image'], gblur_size, crop_right_extra)

    color_crop = get_rotated_image(simple_crop(image, row0, row1, col0, col1), crop1['angle'])[
                 crop1['crop_range'][0]:crop1['crop_range'][1], crop2['crop_range'][0]:crop2['crop_range'][1]]

    sub_ROI = get_sub_ROI(color_crop, row_offset, col_offset, border)

    color_channel = np.array([color_channel]).flatten()  # convert to array, in case
    signal_type = np.array([signal_type]).flatten()  # convert to array, in case
    result = [process_ROI(each_ROI, each_channel, each_type,
                          do_detrend, ndetrend, nsignal,
                          sobel_size, nblur, top_fraction, contour_mode, contour_method, rect_height, edge_gap) for
              each_ROI, each_channel, each_type in zip(sub_ROI['sub_list'], color_channel, signal_type)]

    cropped_image = sub_ROI['image'].copy()

    for i, each_result in enumerate(result):
        if 'hline' in each_result.keys():
            for each in each_result['hline']:
                y_draw = each + row_offset
                if i == 0:
                    x0 = col_offset
                else:
                    x0 = border[i-1]
                x1 = border[i]
                cv.line(cropped_image, (x0, y_draw), (x1, y_draw), (255, 0, 255), 2)

    return {'result': result,
            'cropped_image': cropped_image,
            'col_offset': col_offset}  # return col_offset for convenient plotting


def process_folder(folder, method, output_name):
    '''
    process through png images in a folder
    :param folder: input folder
    :param method: method to process each image
    :param output_name: name of output tif and csv
    :return: none
    '''
    print(folder)

    filelist = [each for each in os.listdir(folder) if each.endswith('.png')]
    # sort by the time stamp
    argsort = np.argsort(pd.Series(filelist).str.split('_', n=1, expand=True).loc[:, 1].values)
    filelist = np.array(filelist)[argsort]
    all_out = [method(cv.imread(os.path.join(folder, each))) for each in filelist]

    all_signal = np.array([[eacheach['signal'] for eacheach in each['result']] for each in all_out])

    # plot
    figx_each = 6
    figy_each = 4
    img_cols = min(8, len(all_signal))
    img_rows = int(np.ceil(len(filelist) / img_cols)) * 2

    # plot ROI
    fig0, axs0 = plt.subplots(img_rows, img_cols, sharex=True, sharey=False,
                              figsize=(figx_each*img_cols, figy_each*img_rows), squeeze=True)
    axs0 = axs0.flatten()

    # make map of indices of subplot from indices of input images
    n_image = len(all_out)
    i_image = np.arange(n_image)
    i_ROI = (np.floor(i_image / img_cols) * 2 * img_cols).astype(int) + i_image % img_cols
    i_line = i_ROI + img_cols

    for i, each in enumerate(all_out):
        each_title = filelist[i] + '; ' + '; '.join(all_signal[i].round().astype(int).astype(str))
        axs0[i_ROI[i]].set_title(each_title)
        axs0[i_ROI[i]].imshow(each['cropped_image'][:, :, ::-1])

        if np.any(np.isnan(np.concatenate([each_result['line'] for each_result in each['result']]))):
            print('image analysis failed for ' + filelist[i])
        else:
            axs0[i_line[i]].set_title(each_title)
            x0 = each['col_offset']
            for eacheach in each['result']:
                y_plot = eacheach['line']
                x_plot = np.arange(x0, x0 + len(y_plot))
                axs0[i_line[i]].plot(x_plot, y_plot)
                x0 = x0 + len(y_plot)
            axs0[i_line[i]].axhline(y=0, color='k')
            axs0[i_line[i]].set_ylabel('signal')

    fig0.savefig(os.path.join(folder, output_name + '_ROI_line.pdf'))
    plt.clf()
    plt.cla()
    plt.close('all')

    signal_df = pd.DataFrame(data=np.hstack([filelist.reshape((-1, 1)), all_signal]),
                             columns=['filename'] + ['signal_' + str(each) for each in range(all_signal.shape[1])])
    signal_df.to_csv(os.path.join(folder, output_name + '.csv'), index=False)


def process_folder_manual_ROI(folder, method, output_name):
    '''
    process through png images in a folder, each of which is already an ROI
    :param folder: input folder
    :param method: method to process each image
    :param output_name: name of output files
    :return: none
    '''
    print(folder)

    filelist = [each for each in os.listdir(folder) if each.endswith('.png')]
    # sort by the time stamp
    argsort = np.argsort(pd.Series(filelist).str.split('_', n=1, expand=True).loc[:, 1].values)
    filelist = np.array(filelist)[argsort]
    all_out = [method(cv.imread(os.path.join(folder, each))) for each in filelist]

    all_signal = np.array([each['signal'] for each in all_out])

    # plot
    figx_each = 6
    figy_each = 4
    img_cols = min(8, len(all_signal))
    img_rows = int(np.ceil(len(filelist) / img_cols)) * 2

    # plot ROI
    fig0, axs0 = plt.subplots(img_rows, img_cols, sharex=True, sharey=False,
                              figsize=(figx_each*img_cols, figy_each*img_rows), squeeze=True)
    axs0 = axs0.flatten()

    # make map of indices of subplot from indices of input images
    n_image = len(all_out)
    i_image = np.arange(n_image)
    i_ROI = (np.floor(i_image / img_cols) * 2 * img_cols).astype(int) + i_image % img_cols
    i_line = i_ROI + img_cols

    for i, each in enumerate(all_out):
        each_title = filelist[i] + '; ' + str(int(round(each['signal'])))
        axs0[i_ROI[i]].set_title(each_title)
        axs0[i_ROI[i]].imshow(each['image_color'][:, :, ::-1])

        axs0[i_line[i]].set_title(each_title)
        axs0[i_line[i]].plot(each['line'])
        axs0[i_line[i]].axhline(y=0, color='k')
        axs0[i_line[i]].set_ylabel('signal')

    fig0.savefig(os.path.join(folder, output_name + '_ROI_line.pdf'))
    plt.clf()
    plt.cla()
    plt.close('all')

    signal_df = pd.DataFrame(data=np.hstack([filelist.reshape((-1, 1)), all_signal.reshape((-1, 1))]),
                             columns=['filename', 'signal'])
    signal_df.to_csv(os.path.join(folder, output_name + '.csv'), index=False)


