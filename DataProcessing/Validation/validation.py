import pandas as pd
import os
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



def fixation_vs_boundingboxes_statistics(dataset_folder):
    '''
    This method is called by t-test_analysis() method and it generates:
     1)  fixations_vs_bounding_boxes.csv containing the number of fixations per image
        per bounding box (e.g. mediastinium, spine, left costophrenic angle, etc.)
     2) fixations_vs_bounding_boxes_time_duration.csv containing accumulated number fixations per condition (i.e Normal, CHF, Pneumonia)
     . This table was used to generate Figure 11 in the paper.

    '''

    #Load bounding boxes spreadsheet
    bbox_table = pd.read_csv(os.path.join(dataset_folder,'bounding_boxes.csv'))

    #Load fixations spreadsheet
    datatype_table = pd.read_csv(os.path.join(dataset_folder,'fixations.csv'))

    #Load master sheet
    cases = pd.read_csv(os.path.join(dataset_folder,'master_sheet.csv'))

    #Number of images per condition
    num_normal_cases = len(cases.loc[cases['Normal'] == 1])
    num_pneumonia_cases = len(cases.loc[cases['pneumonia'] == 1])
    num_chf_cases = len(cases.loc[cases['CHF'] == 1])

    #Get names of bounding boxes
    unique_bbox_names = np.unique(bbox_table['bbox_name'].values)

    #Save visited cases to perform validation that all images were visited at the end
    case_indices = []

    bbox_names_cases = pd.DataFrame(columns=[unique_bbox_names])
    bbox_names_cases['dicom_id'] = cases['dicom_id']
    bbox_names_cases['condition'] = 'empty'
    bbox_names_cases[unique_bbox_names] = 0
    # bbox_names_cases.to_csv('fixations_vs_bounding_boxes.csv', index=False)


    bbox_names_normal_time_duration = {i: 0 for i in unique_bbox_names}
    bbox_names_chf_time_duration = bbox_names_normal_time_duration.copy()
    bbox_names_pneumonia_time_duration = bbox_names_normal_time_duration.copy()



    for index, row in bbox_table.iterrows():
        if index % 100 == 0:
            print("Finished ", index, ' of ', bbox_table.shape[0])
            bbox_names_cases.to_csv('fixations_vs_bounding_boxes.csv', index=False)

        #Dicom image name
        image_name = row['dicom_id']
        #Anatomy name
        bbox_name = row['bbox_name']

        bbox_coordinates = [int(row['x1']), int(row['x2']), int(row['y1']),
                            int(row['y2'])]

        try:
            datatype_case_index_list = datatype_table.index[
                datatype_table['DICOM_ID'] == image_name].tolist()
            case_index = cases.index[cases['dicom_id'] == image_name].tolist()[0]
            case_indices.append(case_index)

            for pointer, i in enumerate(datatype_case_index_list):
                x = datatype_table.loc[i, 'X_ORIGINAL']
                y = datatype_table.loc[i, 'Y_ORIGINAL']

                if pointer == 0:
                    time_duration = datatype_table.loc[i, 'Time (in secs)']
                else:
                    time_duration = datatype_table.loc[i, 'Time (in secs)'] - datatype_table.loc[
                        i - 1, 'Time (in secs)']

                if x >= bbox_coordinates[0] and x <= bbox_coordinates[1] and y >= bbox_coordinates[2] and y <= \
                        bbox_coordinates[3]:
                    bbox_names_cases.loc[case_index, bbox_name] = bbox_names_cases.loc[case_index, bbox_name]+1

                    if cases.loc[case_index, 'Normal'] == 1:
                        bbox_names_cases.loc[case_index, 'condition'] = 'normal'
                        bbox_names_normal_time_duration[bbox_name] = bbox_names_normal_time_duration[
                                                                         bbox_name] + time_duration


                    if cases.loc[case_index, 'CHF'] == 1:
                        bbox_names_cases.loc[case_index, 'condition'] = 'CHF'
                        bbox_names_chf_time_duration[bbox_name] = bbox_names_chf_time_duration[
                                                                      bbox_name] + time_duration



                    if cases.loc[case_index, 'pneumonia'] == 1:
                        bbox_names_cases.loc[case_index, 'condition'] = 'pneumonia'
                        bbox_names_pneumonia_time_duration[bbox_name] = bbox_names_pneumonia_time_duration[
                                                                            bbox_name] + time_duration

        except:
            print('Error: ', image_name)


    #Do validation that all the images were accounted for and we didn't miss any DICOM image
    case_indices = np.unique(np.asarray(case_indices))
    for i in range(len(cases)):
        if i not in case_indices:
            print('Error ', i)


    #Save fixations_vs_bounding_boxes
    bbox_names_cases.to_csv('fixations_vs_bounding_boxes.csv', index=False)


    #Save fixations_vs_bounding_boxes_time_duration
    conditions = ['Normal', 'Pneumonia', 'CHF']

    #Normalize
    for key in bbox_names_normal_time_duration:
        print("before ", key, bbox_names_normal_time_duration[key] )
        bbox_names_normal_time_duration[key] /= num_normal_cases
        bbox_names_pneumonia_time_duration[key] /= num_pneumonia_cases
        bbox_names_chf_time_duration[key] /= num_chf_cases
        print("after ", key, bbox_names_normal_time_duration[key] )

    frames = [pd.DataFrame([bbox_names_normal_time_duration]), pd.DataFrame([bbox_names_pneumonia_time_duration]),
              pd.DataFrame([bbox_names_chf_time_duration])]
    frames = pd.concat(frames)
    frames.insert(0, 'Condition', conditions)
    frames.to_csv('fixations_vs_bounding_boxes_time_duration.csv', index=False)



def transcript_statistics(dataset_folder):
    '''
    This method runs the transcripts validation as described in the Validation section of the paper.
    '''

    print('\n----- TRANSCRIPTS VALIDATION -----')
    subfolders = [f.path for f in
               os.scandir(os.path.join(dataset_folder,'audio_segmentation_transcripts')) if
               f.is_dir()]
    total_single_words = 0
    total_multiple_words = 0
    total_num_instances = 0
    for subfolder in subfolders:
        with open(os.path.join(subfolder, "transcript.json"), "r") as read_file:
            transcript = json.load(read_file)
            phrases = transcript['time_stamped_text']
            for phrase in phrases:
                num_words = len(phrase['phrase'].split(' '))
                if num_words == 1:
                    total_single_words += 1
                else:
                    total_multiple_words += 1
                total_num_instances += 1

    print("Number of instances with single words:  ",total_single_words, "\nNumber instances with multiple phrases: ",total_multiple_words, "\nTotal number of instances: ",total_num_instances, "\nType B Error: ",1-total_single_words / total_num_instances)


def calibration_statistics(dataset_folder):
    '''
    This method runs the validation of eye tracking accuracy as described in the Validation section of the paper
    '''

    print('\n----- CALIBRATION VALIDATION -----')

    cases = pd.read_csv(os.path.join(dataset_folder,'master_sheet.csv'))
    fixation_table = pd.read_csv(os.path.join(dataset_folder,'fixations.csv'))
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




def t_test_analysis(dataset_folder):
    '''
    T-test analysis of fixations between conditions (i.e. pneumonia vs normal, CHF vs normal, pneumonia vs CHF)
    per anatomy bounding box. This is described in detail in the 'Validation' section of the manuscript.
    '''
    print('\n----- T-TEST ANALYSIS -----')

    #Create fixations vs anatomies vs conditions table
    fixation_vs_boundingboxes_statistics(dataset_folder)

    df = pd.read_csv('fixations_vs_bounding_boxes.csv')
    df['condition'].unique()

    comparisons = [['normal', 'pneumonia'], ['normal', 'CHF'], ['pneumonia', 'CHF']]


    # This is a t-test on the total number of fixations
    for cur in comparisons:
        print(cur)
        print(ss.ttest_ind(df.drop(['dicom_id', 'condition'], axis=1)[df['condition'] == cur[0]].sum(axis=1),
                           df.drop(['dicom_id', 'condition'], axis=1)[df['condition'] == cur[1]].sum(axis=1)))


    df2 = df.drop(['dicom_id', 'condition'], axis=1)


    # This is a ttest on the number of fixations for each anatomical structure per condition pair
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
    # draw_bounding_boxes()
    #Replace with the folder you downloaded the eye gaze files
    #Replace with the folder that your MIMIC images are downloaded
    dataset_folder = '../../Resources'

    #Replace with the folder that your MIMIC images are downloaded
    original_folder_images = '/gpfs/fs0/data/mimic_cxr/images/'

    #Run transcript statistics as described in the paper
    transcript_statistics(dataset_folder)

    #Run calibration statistics as described in the paper
    calibration_statistics(dataset_folder)

    #Run eye gaze fixation vs bounding boxes validation as described in the paper
    t_test_analysis(dataset_folder)
