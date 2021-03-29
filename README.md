[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Creation and Validation of a Chest X-Ray Dataset with Eye-tracking and Report Dictation for AI Tool Development

### Introduction
This repository contains the code to reproduce the experiments and data preparation for the paper **"Creation and Validation of a Chest X-Ray Dataset with Eye-tracking and Report Dictation for AI Tool Development"**.
If you find this dataset, models or code useful, please cite us using the following bibTex:
```
%Paper:
@article{karargyris2021creation,
  title={Creation and validation of a chest X-ray dataset with eye-tracking and report dictation for AI development},
  author={Karargyris, Alexandros and Kashyap, Satyananda and Lourentzou, Ismini and Wu, Joy T and Sharma, Arjun and Tong, Matthew and Abedin, Shafiq and Beymer, David and Mukherjee, Vandana and Krupinski, Elizabeth A and others},
  journal={Scientific Data},
  volume={8},
  number={1},
  pages={1--18},
  year={2021},
  publisher={Nature Publishing Group}
}

%Dataset:
@misc{Karargyris2020,
author = {Karargyris, Alexandros and Kashyap, Satyananda and Lourentzou, Ismini and Wu, Joy and Tong, Matthew and Sharma, Arjun and Abedin, Shafiq and Beymer, David and Mukherjee, Vandana and Krupinski, Elizabeth and Moradi, Mehdi},
booktitle = {Physionet},
doi = {https://doi.org/10.13026/qfdz-zr67},
title = {{Eye Gaze Data for Chest X-rays (version 1.0.0)}},
url = {https://physionet.org/content/egd-cxr/1.0.0/},
year = {2020}
}
```
A high level summary of the work can be found [here](https://researchdata.springernature.com/posts/creation-and-validation-of-a-chest-x-ray-dataset-with-eye-tracking-and-report-dictation-for-ai-development?channel_id=behind-the-paper) 

### Clone this repo by typing in the command line:
```bash 
git clone https://github.com/cxr-eye-gaze/eye-gaze-dataset.git
cd eye-gaze-dataset/
```

### Download the dataset
To access the datasets, sign the user agreements for [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) and for the [Eye Gaze Data](https://physionet.org/content/egd-cxr/1.0.0/)

Then, download MIMIC-CXR dataset
```bash
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/
```
Also, download our dataset:
```bash
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/files/egd-cxr/1.0.0/
```
where **USERNAME** is your physionet.org username (the commands will prompt for user password). 


### Repository Structure:
- [Data Processing folder](./DataProcessing) contains code to post process the data (i.e. map eye gaze coordinates to image coordinates, run speech to text on audio files), prepare the master_sheet.csv file and images used in the study, and reproduce validations as described in the paper.
Read [the readme file](./DataProcessing/readme.md) for more details. 
- [Experiments](./Experiments) contains code for the machine learning experiments presented in the paper.
