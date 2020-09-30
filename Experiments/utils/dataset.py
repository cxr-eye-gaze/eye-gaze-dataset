# -- Generator
import os
import re
import glob
import torch
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from torch.utils.data import Dataset


def split_dataset(file_path, random_state=1):
    df = pd.read_csv(file_path)
    # -- Split after uniquing the patient ids so that it does not get split across the different test, dev, test
    pid = list(df['patient_id'].unique())
    random.seed(random_state)
    random.shuffle(pid)
    train_patient_count = round(len(pid) * 0.8)
    not_train = len(pid) - train_patient_count
    # --- Split this remaining equally into dev and test.
    dev_patient_count = round(not_train * 0.5)
    train = df[df['patient_id'].isin(pid[:train_patient_count])]
    dev = df[df['patient_id'].isin(pid[train_patient_count:train_patient_count+dev_patient_count])]
    test = df[df['patient_id'].isin(pid[train_patient_count+dev_patient_count:])]
    return train, dev, test


def read_dicoms(image_path):
    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)
    image = image/np.max(image)
    image = np.moveaxis(image, 0, -1)
    return image


def collate_fn(data):
    """
    Custom collate_fn dynamic padding, sort by sequence length (descending order)
    Sequences are padded to the maximum length of mini-batch sequences.
    """
    data.sort(key=lambda x: len(x[-2]), reverse=True)
    image, y_label, idx, X_hm, y_hm = zip(*data)
    image = torch.stack(image, dim=0)
    y_label = torch.stack(y_label, dim=0)
    y_hm = torch.stack(y_hm, dim=0)
    if isinstance(X_hm[0], torch.Tensor): X_hm = torch.nn.utils.rnn.pad_sequence(X_hm, batch_first=True)
    return image, y_label, idx, X_hm, y_hm


class EyegazeDataset(Dataset):
    def __init__(self, csv_file, image_path_name, class_names, static_heatmap_path=None, heatmaps_path=None, heatmap_temporal_transform=None,
                                       heatmap_static_transform=None, image_transform=None, heatmaps_threshold=None):
        self.csv_file = csv_file
        self.path_name = image_path_name
        self.image_transform = image_transform
        self.heatmap_static_transform = heatmap_static_transform
        self.heatmap_temporal_transform = heatmap_temporal_transform
        self.class_names = class_names
        self.heatmaps_path = heatmaps_path
        self.static_heatmap_path = static_heatmap_path
        self.heatmaps_threshold = heatmaps_threshold

    def __len__(self):
        return len(self.csv_file)

    def get_image(self, idx):
        # -- Query the index location of the required file
        image_path = os.path.join(self.path_name,  self.csv_file['path'].iloc[idx])
        image = read_dicoms(image_path)
        if len(image.shape) == 2: image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)
        truth_labels = [self.csv_file[labels].iloc[idx] for labels in self.class_names]
        y_label = np.array(truth_labels, dtype=np.int64).tolist()
        if self.image_transform:
            image = self.image_transform(image)
        return image.float(), torch.from_numpy(np.array(y_label)).float()

    def num_sort(self, filename):
        not_num = re.compile("\D")
        return int(not_num.sub("", filename))

    def __getitem__(self, idx):
        image_name = self.csv_file['dicom_id'].iloc[idx]
        image, y_label = self.get_image(idx)
        X_hm, y_hm = [], []
        if self.heatmaps_path:
            heat_path = os.path.join(self.heatmaps_path, image_name)
            for im in sorted(glob.glob(heat_path + '/*frame.png'), key=self.num_sort):
                img = Image.open(im).convert('RGB')
                img = self.heatmap_temporal_transform(img)
                if self.heatmaps_threshold:
                    img = img > self.heatmaps_threshold
                X_hm.append(img)
            if len(X_hm) == 0:
                raise FileNotFoundError(f'temporal heatmaps not found for {heat_path}')
            X_hm = torch.stack(X_hm, dim=0)
        if self.static_heatmap_path:
            heat_path = os.path.join(self.static_heatmap_path, image_name)
            # ground_truth static heatmap
            if not os.path.exists(heat_path + '/heatmap.png'):
                raise FileNotFoundError(f'static heatmaps not found for {heat_path}')
            y_hm = Image.open(heat_path + '/heatmap.png').convert('RGB')
            y_hm = self.heatmap_static_transform(y_hm)
            if self.heatmaps_threshold:
                y_hm = y_hm > self.heatmaps_threshold
        return image, y_label, idx, X_hm, y_hm
