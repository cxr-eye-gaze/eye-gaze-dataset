import pandas as pd
import os
from scipy import ndimage
import scipy
from matplotlib import image
import pydicom
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
import scipy.stats as ss


def crop(image):
    '''
    Auxilary function to crop image to non-zero area
    :param image: input image
    :return: cropped image
    '''
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]



def bbox_eye_gaze_statistics():
    bbox_table = pd.read_csv(
        'mapped_bbox_table.csv')
    datatype_table = pd.read_csv(
        '/Users/akararg@us.ibm.com/PycharmProjects/eye-gaze-ai/GenerateEyeGazeVideo/all_fixations.csv')
    cases = pd.read_csv('master_sheet.csv')
    unique_bbox_names = np.unique(bbox_table['bbox_name'].values)
    case_indices = []

    bbox_names_cases = pd.DataFrame(columns=[unique_bbox_names])
    bbox_names_cases['dicom_id'] = cases['dicom_id']
    bbox_names_cases['condition'] = ''
    bbox_names_cases[unique_bbox_names] = 0
    bbox_names_cases.to_csv('/Users/akararg@us.ibm.com/PycharmProjects/eye-gaze-ai/FOR_RELEASE/bbox_full_cases.csv', index=False)


    bbox_names_normal = {i: 0 for i in unique_bbox_names}
    bbox_names_chf = bbox_names_normal.copy()
    bbox_names_pneumonia = bbox_names_normal.copy()

    bbox_names_normal_time_duration = {i: 0 for i in unique_bbox_names}
    bbox_names_chf_time_duration = bbox_names_normal.copy()
    bbox_names_pneumonia_time_duration = bbox_names_normal.copy()

    bbox_names_normal_25 = {i: 0 for i in unique_bbox_names}
    bbox_names_chf_25 = bbox_names_normal.copy()
    bbox_names_pneumonia_25 = bbox_names_normal.copy()

    bbox_names_normal_50 = {i: 0 for i in unique_bbox_names}
    bbox_names_chf_50 = bbox_names_normal.copy()
    bbox_names_pneumonia_50 = bbox_names_normal.copy()

    bbox_names_normal_75 = {i: 0 for i in unique_bbox_names}
    bbox_names_chf_75 = bbox_names_normal.copy()
    bbox_names_pneumonia_75 = bbox_names_normal.copy()

    bbox_names_normal_100 = {i: 0 for i in unique_bbox_names}
    bbox_names_chf_100 = bbox_names_normal.copy()
    bbox_names_pneumonia_100 = bbox_names_normal.copy()

    for index, row in bbox_table.iterrows():
        if index % 100 == 0:
            print("Finished ", index, ' of ', bbox_table.shape[0])
        image_name = row['dicom_id'].split('.')[0]
        bbox_name = row['bbox_name']
        bbox_coordinates = [int(row['x1']), int(row['x2']), int(row['y1']),
                            int(row['y2'])]

        try:
            datatype_case_index_list = datatype_table.index[
                datatype_table['MEDIA_NAME'] == image_name + '.png'].tolist()
            case_index = cases.index[cases['dicom_id'] == image_name].tolist()[0]
            case_indices.append(case_index)

            first_row = datatype_case_index_list[0]
            last_row = datatype_case_index_list[-1]
            length = last_row - first_row
            length_25 = [first_row, first_row + length // 4]
            length_50 = [first_row + length // 4, first_row + length // 2]
            length_75 = [first_row + length // 2, last_row - length // 4]
            length_100 = [last_row - length // 4, last_row]

            for pointer, i in enumerate(datatype_case_index_list):
                x = datatype_table.loc[i, 'X_ORIGINAL']
                y = datatype_table.loc[i, 'Y_ORIGINAL']

                if pointer == 0:
                    time_duration = datatype_table.loc[i, 'TIME(in seconds)']
                else:
                    time_duration = datatype_table.loc[i, 'TIME(in seconds)'] - datatype_table.loc[
                        i - 1, 'TIME(in seconds)']

                if x >= bbox_coordinates[0] and x <= bbox_coordinates[1] and y >= bbox_coordinates[2] and y <= \
                        bbox_coordinates[3]:
                    bbox_names_cases.loc[case_index, bbox_name] = bbox_names_cases.loc[case_index, bbox_name]+1

                    if cases.loc[case_index, 'Normal'] == 1:
                        bbox_names_cases.loc[case_index, 'condition'] = 'normal'
                        bbox_names_normal[bbox_name] = bbox_names_normal[bbox_name] + 1
                        bbox_names_normal_time_duration[bbox_name] = bbox_names_normal_time_duration[
                                                                         bbox_name] + time_duration
                        if i < length_25[1]:
                            bbox_names_normal_25[bbox_name] = bbox_names_normal_25[bbox_name] + 1
                        if i >= length_25[1] and i < length_50[1]:
                            bbox_names_normal_50[bbox_name] = bbox_names_normal_50[bbox_name] + 1
                        if i >= length_50[1] and i < length_75[1]:
                            bbox_names_normal_75[bbox_name] = bbox_names_normal_75[bbox_name] + 1
                        if i >= length_75[1] and i <= length_100[1]:
                            bbox_names_normal_100[bbox_name] = bbox_names_normal_100[bbox_name] + 1

                    if cases.loc[case_index, 'CHF'] == 1:
                        bbox_names_cases.loc[case_index, 'condition'] = 'CHF'

                        bbox_names_chf[bbox_name] = bbox_names_chf[bbox_name] + 1
                        bbox_names_chf_time_duration[bbox_name] = bbox_names_chf_time_duration[
                                                                      bbox_name] + time_duration

                        if i < length_25[1]:
                            bbox_names_chf_25[bbox_name] = bbox_names_chf_25[bbox_name] + 1
                        if i >= length_25[1] and i < length_50[1]:
                            bbox_names_chf_50[bbox_name] = bbox_names_chf_50[bbox_name] + 1
                        if i >= length_50[1] and i < length_75[1]:
                            bbox_names_chf_75[bbox_name] = bbox_names_chf_75[bbox_name] + 1
                        if i >= length_75[1] and i <= length_100[1]:
                            bbox_names_chf_100[bbox_name] = bbox_names_chf_100[bbox_name] + 1

                    if cases.loc[case_index, 'pneumonia'] == 1:
                        bbox_names_cases.loc[case_index, 'condition'] = 'pneumonia'

                        bbox_names_pneumonia[bbox_name] = bbox_names_pneumonia[bbox_name] + 1
                        bbox_names_pneumonia_time_duration[bbox_name] = bbox_names_pneumonia_time_duration[
                                                                            bbox_name] + time_duration

                        if i < length_25[1]:
                            bbox_names_pneumonia_25[bbox_name] = bbox_names_pneumonia_25[bbox_name] + 1
                        if i >= length_25[1] and i < length_50[1]:
                            bbox_names_pneumonia_50[bbox_name] = bbox_names_pneumonia_50[bbox_name] + 1
                        if i >= length_50[1] and i < length_75[1]:
                            bbox_names_pneumonia_75[bbox_name] = bbox_names_pneumonia_75[bbox_name] + 1
                        if i >= length_75[1] and i <= length_100[1]:
                            bbox_names_pneumonia_100[bbox_name] = bbox_names_pneumonia_100[bbox_name] + 1
        except:
            print(image_name)

    case_indices = np.unique(np.asarray(case_indices))
    for i in range(len(cases)):
        if i not in case_indices:
            print('Error ', i)

    bbox_names_cases.to_csv('bbox_full_cases.csv', index=False)
    conditions = ['Normal', 'Pneumonia', 'CHF']

    frames = [pd.DataFrame([bbox_names_normal_time_duration]), pd.DataFrame([bbox_names_pneumonia_time_duration]),
              pd.DataFrame([bbox_names_chf_time_duration])]
    frames = pd.concat(frames)
    frames.insert(0, 'Condition', conditions)
    frames.to_csv('bboxes_time_duration.csv', index=False)

    frames = [pd.DataFrame([bbox_names_normal]), pd.DataFrame([bbox_names_pneumonia]), pd.DataFrame([bbox_names_chf])]
    frames = pd.concat(frames)
    frames.insert(0, 'Condition', conditions)
    frames.to_csv('fixation_anatomies.csv', index=False)

    frames = [pd.DataFrame([bbox_names_normal_25]), pd.DataFrame([bbox_names_pneumonia_25]),
              pd.DataFrame([bbox_names_chf_25])]
    frames = pd.concat(frames)
    frames.insert(0, 'Condition', conditions)
    frames.to_csv('fixation_anatomies_25.csv', index=False)

    frames = [pd.DataFrame([bbox_names_normal_50]), pd.DataFrame([bbox_names_pneumonia_50]),
              pd.DataFrame([bbox_names_chf_50])]
    frames = pd.concat(frames)
    frames.insert(0, 'Condition', conditions)
    frames.to_csv('fixation_anatomies_50.csv', index=False)

    frames = [pd.DataFrame([bbox_names_normal_75]), pd.DataFrame([bbox_names_pneumonia_75]),
              pd.DataFrame([bbox_names_chf_75])]
    frames = pd.concat(frames)
    frames.insert(0, 'Condition', conditions)
    frames.to_csv('fixation_anatomies_75.csv', index=False)

    frames = [pd.DataFrame([bbox_names_normal_100]), pd.DataFrame([bbox_names_pneumonia_100]),
              pd.DataFrame([bbox_names_chf_100])]
    frames = pd.concat(frames)
    frames.insert(0, 'Condition', conditions)
    frames.to_csv('fixation_anatomies_100.csv', index=False)


def transcript_statistics():
    folders = [f.path for f in
               os.scandir('/Users/akararg@us.ibm.com/PycharmProjects/eye-gaze-ai/SpeechToText/new_transcripts') if
               f.is_dir()]
    total_single_words = 0
    total_multiple_words = 0
    total = 0
    for folder in folders:
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        for subfolder in subfolders:
            with open(os.path.join(subfolder, "transcript.json"), "r") as read_file:
                print(subfolder)
                transcript = json.load(read_file)
                words = []
                for k, v in transcript.items():
                    if 'sentence' in k:
                        for tag in v:
                            if '?' in tag['word']:
                                print(tag['word'])
                            num_words = len(tag['word'].split(' '))
                            if num_words == 1:
                                total_single_words += 1
                            else:
                                total_multiple_words += 1
                            total += 1
                            words.append(tag['word'])

    print(total_single_words, total_multiple_words, total, total_single_words / total)


def calibration_statistics():
    cases = pd.read_csv('./master_sheet.csv')
    fixation_table = pd.read_csv('./final_fixations.csv')
    coordinateX_list = []
    coordinateY_list = []
    screen_width = 1920
    screen_height = 1080
    num_points = 0
    previous_name = ''
    for index, row in fixation_table.iterrows():
        value = row['DICOM_ID']
        # found = cases[cases['dicom_id'].str.contains(value)]
        found = cases.index[cases['dicom_id'] == value].tolist()
        if len(found) == 0 and previous_name != value:
            last_row = fixation_table.index[fixation_table['DICOM_ID'] == value].tolist()[-1]
            coordX = fixation_table.loc[last_row, 'FPOGX']
            coordY = fixation_table.loc[last_row, 'FPOGY']
            coordinateX_list.append(abs(0.5 - coordX))
            coordinateY_list.append(abs(0.5 - coordY))
            num_points += 1
        previous_name = value

    meanX = np.mean(coordinateX_list)
    stdX = np.std(coordinateX_list)

    meanY = np.mean(coordinateY_list)
    stdY = np.std(coordinateY_list)

    print("Total calibration images: ", num_points)
    print("Percentage mean error: (%.4f , %.4f), with std: (%.4f, %.4f)" % (meanX, meanY, stdX, stdY))
    print("Pixels mean error: (%.4f , %.4f), with std: (%.4f, %.4f)" % (
    meanX * screen_width, meanY * screen_height, stdX * screen_width, stdY * screen_height))


def percentile_statistics(cases):


    original_folder_images = '/gpfs/fs0/data/mimic_cxr/images/'
    conditions = ["Normal", "CHF", "pneumonia"]
    gazes_table = pd.read_csv('all_fixations.csv')
    try:
        os.mkdir('./statistics')
    except:
        pass

    for condition in conditions:
        canvas25 = np.zeros((3500, 3500))
        canvas50 = np.zeros((3500, 3500))
        canvas75 = np.zeros((3500, 3500))
        canvas100 = np.zeros((3500, 3500))

        condition_rows = cases.index[cases[condition] == 1]
        condition_table = cases.loc[condition_rows]
        values = (condition_table['dicom_id'].values + '.png').tolist()
        unique_list = (list(set(values)))
        whole_image = np.zeros((3500, 3500), dtype=np.float)
        for imagename in unique_list:
            ds = pydicom.dcmread(os.path.join(original_folder_images, imagename.split('.')[0] + '.dcm'))
            dicom_image = ds.pixel_array.copy().astype(np.float)
            dicom_image /= np.max(dicom_image)
            dicom_image *= 255.
            dicom_image = dicom_image.astype(np.uint8)
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # print(image.shape)
            whole_image += cv2.resize(dicom_image, (3500, 3500))
        whole_image /= np.max(whole_image)
        whole_image *= 255
        cv2.imwrite('./statistics/' + condition + '.png', whole_image)
        for value in unique_list:

            first_row = gazes_table.index[gazes_table['MEDIA_NAME'] == value].tolist()[0]
            last_row = gazes_table.index[gazes_table['MEDIA_NAME'] == value].tolist()[-1]
            length = last_row - first_row
            length_25 = [first_row, first_row + length // 4]
            length_50 = [first_row + length // 4, first_row + length // 2]
            length_75 = [first_row + length // 2, last_row - length // 4]
            length_100 = [last_row - length // 4, last_row]
            # print(length_25,' ',length_50,' ', length_75, ' ', length_100 )
            for index in range(length_25[0], length_25[1]):
                x_original = gazes_table['X_ORIGINAL'][index]
                y_original = gazes_table['Y_ORIGINAL'][index]
                canvas25[y_original, x_original] += 1

            for index in range(length_50[0], length_50[1]):
                x_original = gazes_table['X_ORIGINAL'][index]
                y_original = gazes_table['Y_ORIGINAL'][index]
                canvas50[y_original, x_original] += 1

            for index in range(length_75[0], length_75[1]):
                x_original = gazes_table['X_ORIGINAL'][index]
                y_original = gazes_table['Y_ORIGINAL'][index]
                canvas75[y_original, x_original] += 1

            for index in range(length_100[0], length_100[1]):
                x_original = gazes_table['X_ORIGINAL'][index]
                y_original = gazes_table['Y_ORIGINAL'][index]
                canvas100[y_original, x_original] += 1

        canvas25[2500:3500, :] = 0
        canvas25[:, 2500:3500] = 0

        canvas50[2500:3500, :] = 0
        canvas50[:, 2500:3500] = 0

        canvas75[2500:3500, :] = 0
        canvas75[:, 2500:3500] = 0

        canvas100[2500:3500, :] = 0
        canvas100[:, 2500:3500] = 0

        heatmap = scipy.ndimage.gaussian_filter(canvas25, 50)
        heatmap = crop(heatmap)
        heatmap = cv2.resize(heatmap, (3500, 3500))
        # image.imsave('./statistics/0-25_'+condition+'.png', heatmap)
        plt.figure()
        plt.imshow(whole_image, 'gray', interpolation='none')
        plt.imshow(heatmap, 'jet', interpolation='none', alpha=0.25)
        plt.axis('off')
        plt.savefig('./statistics/0-25_' + condition + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        heatmap = scipy.ndimage.gaussian_filter(canvas50, 50)
        heatmap = crop(heatmap)
        heatmap = cv2.resize(heatmap, (3500, 3500))

        # image.imsave('./statistics/25-50_'+condition+'.png', heatmap)
        plt.figure()
        plt.imshow(whole_image, 'gray', interpolation='none')
        plt.imshow(heatmap, 'jet', interpolation='none', alpha=0.25)
        plt.axis('off')
        plt.savefig('./statistics/25-50_' + condition + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        heatmap = scipy.ndimage.gaussian_filter(canvas75, 50)
        heatmap = crop(heatmap)
        heatmap = cv2.resize(heatmap, (3500, 3500))

        # image.imsave('./statistics/50_75'+condition+'.png', heatmap)
        plt.figure()
        plt.imshow(whole_image, 'gray', interpolation='none')
        plt.imshow(heatmap, 'jet', interpolation='none', alpha=0.25)
        plt.axis('off')
        plt.savefig('./statistics/50_75_' + condition + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        heatmap = scipy.ndimage.gaussian_filter(canvas100, 50)
        heatmap = crop(heatmap)
        heatmap = cv2.resize(heatmap, (3500, 3500))

        # image.imsave('./statistics/75_100'+condition+'.png', heatmap)
        plt.figure()
        plt.imshow(whole_image, 'gray', interpolation='none')
        plt.imshow(heatmap, 'jet', interpolation='none', alpha=0.25)
        plt.axis('off')
        plt.savefig('./statistics/75_100_' + condition + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()


def statistics(subfolders, data_type, cases):
    conditions = ["Normal", "CHF", "pneumonia"]
    for i, subfolder in enumerate(subfolders):
        if i == 0:
            whole_table = pd.read_csv(os.path.join(subfolder, data_type + '.csv'))
        else:
            gazes_table = pd.read_csv(os.path.join(subfolder, data_type + '.csv'))
            whole_table = pd.concat([whole_table, gazes_table], ignore_index=True)

    for condition in conditions:
        canvas = np.zeros((3500, 3500))

        condition_rows = cases.index[cases[condition] == 1]
        condition_table = cases.loc[condition_rows]
        values = (condition_table['dicom_id'].values + '.png').tolist()

        final_table = whole_table[whole_table['MEDIA_NAME'].isin(values)]
        x_original = final_table["X_ORIGINAL"]
        y_original = final_table["Y_ORIGINAL"]
        for i, j in zip(y_original, x_original):
            canvas[i, j] += 1
        canvas[2500:3500, :] = 0
        canvas[:, 2500:3500] = 0
        # canvas /= np.max(canvas)
        # canvas *= 100
        heatmap = scipy.ndimage.gaussian_filter(canvas, 50)
        heatmap = crop(heatmap)
        image.imsave(condition + '.png', heatmap)

def t_test_analysis():
    
    #Create fixations vs anatomies vs conditions table
    bbox_eye_gaze_statistics()

    df = pd.read_csv('bbox_full_cases.csv')
    df['condition'].unique()

    comparisons = [['normal', 'pneumonia'], ['normal', 'CHF'], ['pneumonia', 'CHF']]


    # This is a t-test on the total number of fixations
    for cur in comparisons:
        print(cur)
        print(ss.ttest_ind(df.drop(['dicom_id', 'condition'], axis=1)[df['condition'] == cur[0]].sum(axis=1),
                           df.drop(['dicom_id', 'condition'], axis=1)[df['condition'] == cur[1]].sum(axis=1)))


    df2 = df.drop(['dicom_id', 'condition'], axis=1)


    # This is a ttest on the number of fixations for each anatomical structure
    res = dict()
    for cur_col in df2.columns:
        print(cur_col)
        res[cur_col] = dict()
        for cur_cond in comparisons:
            comp_lab = cur_cond[0] + ' vs ' + cur_cond[1]
            print(comp_lab)
            print(
                ss.ttest_ind(df[df['condition'] == cur_cond[0]][cur_col], df[df['condition'] == cur_cond[1]][cur_col]))
            (t, p) = ss.ttest_ind(df[df['condition'] == cur_cond[0]][cur_col],
                                  df[df['condition'] == cur_cond[1]][cur_col])
            res[cur_col][comp_lab] = p
        print()
    resout = pd.DataFrame(res)


    resout.to_csv('fixations_vs_anatomy_vs_condition.csv')



if __name__ == "__main__":
    t_test_analysis()
    # import matplotlib.pyplot as plt
    # import pandas as pd
    #
    # values = np.loadtxt('/Users/akararg@us.ibm.com/Downloads/untitled.txt',delimiter=',')
    # bins=[0, 20, 40, 50, 60, 80, 100, 120, 140, 160]
    # # An "interface" to matplotlib.axes.Axes.hist() method
    # n, bins, patches = plt.hist(x=values, bins=bins, color='#0504aa',
    #                             alpha=0.7, rwidth=0.85)
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Frames')
    # plt.text(23, 45, r'$\mu=15, b=3$')
    # maxfreq = n.max()
    # # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    # plt.savefig('frequency.png')
    # subfolders = [f.path for f in os.scandir('/home/akararg/Desktop/MyProjects/eye-gaze-ai/GenerateEyeGazeVideo/sessions') if f.is_dir()]

    # cases = pd.read_csv('./master_sheet.csv')
    # percentile_statistics(cases)
    # statistics(subfolders,'fixations',cases)
