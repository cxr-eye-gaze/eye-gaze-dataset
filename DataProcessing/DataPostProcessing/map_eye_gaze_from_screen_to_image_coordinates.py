import pandas as pd
import multiprocessing
import pydicom
import cv2
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")


#Replace with your dicom_id_folder as described in readme file
original_folder_images='/gpfs/fs0/data/mimic_cxr/images/'

def calibrate(eye_gaze_table,screen_width=1920, screen_height=1080):
    '''
    This method uses calibration image (read paper for more) and recalibrates coordinates

    :param gaze_table: pandas dataframe with eye gaze table
    :param screen_width: DO NOT CHANGE. This was used in the original eye gaze experiment.
    :param screen_height: DO NOT CHANGE. This was used in the original eye gaze experiment.
    :return:
    '''
    try:

        calibrationX=[]
        calibrationY=[]

        # Iterate through each image in the raw eye gaze spreadsheet
        for index, row in eye_gaze_table.iterrows():
            image_name = row['DICOM_ID']

            if os.path.exists(os.path.join(original_folder_images, image_name + '.dcm')) == False:
                last_row = eye_gaze_table.index[eye_gaze_table['DICOM_ID'] == image_name ].tolist()[-1]
                eyeX = eye_gaze_table['FPOGX'][last_row]* screen_width
                eyeY = eye_gaze_table['FPOGY'][last_row]* screen_height

                # Get pixel coordinates from raw eye gaze coordinates
                # eyeX = row['FPOGX'] * screen_width
                # eyeY = row['FPOGY'] * screen_height
                calibrationX.append(eyeX)
                calibrationY.append(eyeY)

        calibrationX = np.asarray(calibrationX)
        calibrationY = np.asarray(calibrationY)

        mean_X = np.mean(calibrationX)
        mean_Y = np.mean(calibrationY)

        calibratedX = screen_width//2 - mean_X
        calibratedY = screen_height//2 - mean_Y

        return calibratedX, calibratedY
    except:
        print('No calibration available')
        return .0,.0

def map_eye_gaze_to_image_coordinates(session_table, cases, calibration=False, screen_width=1920, screen_height=1080):
    '''
    Method to map eye gaze coordinates to original image coordinates
    :param session_table: the eye gaze sheet (i.e. fixations.csv, eye_gaze.csv) in pandas dataframe
    :param cases: the master sheet in pandas dataframe
    :param calibration: boolean flag for doing calibration or not
    :param screen_width: screen width in pixels for monitor's resolution used in experiment. DO NOT CHANGE!
    :param screen_height: screen height in pixels for monitor's resolution used in experiment. DO NOT CHANGE!
    :return: processed pandas dataframe with X_ORIGINAL, Y_ORIGINAL columns (i.e. eye gaze mapped coordinated to original image)
    '''
    session_table["X_ORIGINAL"] = ""
    session_table["Y_ORIGINAL"] = ""
    # Do calibration
    if calibration:
        calibratedX, calibratedY = calibrate(session_table)
    else:
        calibratedX = calibratedY = .0
    # Iterate through each image in the eye gaze spreadsheet
    for index, row in session_table.iterrows():
        if index%1000==0:
            print('Finished ', index, '/', session_table.shape[0], ' rows from session ',
                  session_table['SESSION_ID'].values[0])

        # Get pixel coordinates from raw eye gaze coordinates and calibrate them
        eyeX = row['FPOGX'] * screen_width + calibratedX
        eyeY = row['FPOGY'] * screen_height + calibratedY
        image_name = row['DICOM_ID']

        try:
            ds = pydicom.dcmread(os.path.join(original_folder_images, image_name + '.dcm'))

            image = ds.pixel_array.copy().astype(np.float)
            image /= np.max(image)
            image *= 255.
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            case_index = cases.loc[cases['dicom_id'] == image_name].index[0]
            top, bottom, left, right = cases.loc[case_index, 'image_top'], cases.loc[case_index, 'image_bottom'], \
                                              cases.loc[case_index, 'image_left'], cases.loc[case_index, 'image_right']

        except:
            image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            top, bottom, left, right = (0, 0, 0, 0)
        if (image.shape[0]%2)>0:
            image = np.vstack((image, np.zeros((1, image.shape[1], 3))))
        if (image.shape[1]%2)>0:
            image = np.hstack((image, np.zeros((image.shape[0], 1, 3))))
        # Keep eye gazes that fall within the original image size, else assign negative values them
        if eyeX > left and eyeX < screen_width - right and eyeY > top and eyeY < screen_height - bottom:
            x_original = eyeX - left
            y_original = eyeY - top
        else:
            x_original = -1
            y_original = -1

        # Remap to original image coordinates
        resized_width = screen_width - left - right
        resized_height = screen_height - top - bottom
        x_original_in_pixels = int((image.shape[1] / resized_width) * x_original)
        y_original_in_pixels = int((image.shape[0] / resized_height) * y_original)


        session_table.loc[index, "X_ORIGINAL"] = x_original_in_pixels
        session_table.loc[index, "Y_ORIGINAL"] = y_original_in_pixels
    return session_table

def concatenate_session_tables(eye_gaze_session_tables):
    '''
    Auxilary method that simply concatenates each individual session eye gaze table into a single table
    :param tables: List of Pandas dataframes of session eye gaze tables
    :return:
    '''
    final_table = []

    for i, table in enumerate(eye_gaze_session_tables):
        if i == 0:
            n_columns = len(table.columns)
            columns = table.columns
            table.columns = range(n_columns)
            final_table = table
        else:
            table.columns = range(n_columns)
            final_table = pd.concat([final_table, table], axis=0,ignore_index=True,sort=False)
    final_table.columns=columns
    return final_table


def process_mapping(experiment_name, datatype):
    print('--------> MAP EYE GAZE TO IMAGE COORDINATES <--------')

    cases = pd.read_csv('master_sheet.csv')
    table = pd.read_csv(datatype)

    #Group by session id to parallelize jobs
    sessions = table.groupby(['SESSION_ID'])

    try:
        os.mkdir(experiment_name)
    except OSError as exc:
        print(exc, ' Proceeding...')
        pass

    p = multiprocessing.Pool(processes=len(sessions))
    objects = []
    for session in sessions:
        df = session[1].copy().reset_index(drop=True)
        objects.append((df, cases))
    eye_gaze_session_tables = p.starmap(map_eye_gaze_to_image_coordinates, [i for i in objects])
    p.close()

    final_table = concatenate_session_tables(eye_gaze_session_tables)

    # Save experiment consolidated table
    final_table.to_csv(experiment_name + '.csv', index=False)

if __name__ == '__main__':
    # To map fixation.csv eye gaze coordinates into original image coordinates uncomment the following line
    process_mapping(experiment_name='fixation_mapping', datatype='fixations.csv')

    # To map eye_gaze.csv eye gaze coordinates into original image coordinates uncomment the following line
    process_mapping(experiment_name='eye_gaze_mapping', datatype='eye_gaze.csv')