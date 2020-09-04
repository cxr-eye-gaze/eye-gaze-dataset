# Creation and Validation of a Chest X-Ray Dataset with Eye-tracking and Report Dictation for AI Tool Development

### Introduction
This repository contains the code to reproduce the experiments and data preparation for the paper "Creation and Validation of a Chest X-Ray Dataset with Eye-tracking and Report Dictation for AI Tool Development"

Specifically:
- [Data Processing folder](./DataProcessing) contains code to post process the data (i.e. map eye gaze coordinates to image coordinates, run speech to text on audio files), prepare the master_sheet.csv file and images used in the study, and reproduce validations as described in the paper.
User should read [the readme file](./DataProcessing/readme.md) for more details. 
- [Experiments](./Experiments) contains code for the machine learning experiments presented in the paper
### Requirements
Download [Eye Gaze dataset from PhysioNet]() and place files in [Resources](/Resources) folder
