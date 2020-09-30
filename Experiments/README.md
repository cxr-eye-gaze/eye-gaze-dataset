# Eyegaze Dataset

This is the code repo for the experiments presented in **"Creation and Validation of a Chest X-Ray Datasetwith Eye-tracking and Report Dictation for AI ToolDevelopment"**. Please follow the [instructions](../DataProcessing/readme.md) in DataProcessing and edit the ``data_path``, ``image_path`` and ``heatmaps_path`` in [main](main.py) and [main_Unet.py](main_Unet.py) accordingly.

### Requirements
The python version is ```python3.6.5```. Package requirements can be installed with one bash command:
```bash
pip3 install -r requirements.txt
```

### Temporal Heatmaps Experiment
i) Train a baseline model, save results in folder `results_baseline` 
```bash
python3 main.py --output_dir results_baseline --epochs 20 --model_type baseline --scheduler --batch_size 16 --dropout 0.5 --hidden_dim 64 --hidden_hm 128 --gpus 3,4,5,6,7
```
ii) Train a temporal heatmaps model (Figure 13 in paper), save results in folder `results_temporal`
```bash
python3 main.py  --output_dir results_temporal --epochs 20 --model_type temporal --scheduler --attention --brnn_hm --batch_size 16 --dropout 0.5 --hidden_dim 64 --hidden_hm 128 --gpus 3,4,5,6,7
```

### Static Heatmaps Experiment

These parameters are taken from the best performing hyperparameter search done by using the tune library. 
To run another hyper-parameter search run `tune_static.py`. Also see  `eye-gaze-results.pptx` for details 
on the different hyper parameter searches and the best tuning experiment result ROC plots.  

i) Train a baseline model, save results in folder `results`.  
```bash
python3 main_Unet.py --model_type baseline --dropout=0.0 --epochs 20 --gamma 1 --lr 0.006486 --model_teacher timm-efficientnet-b0 --step_size 8 --scheduler --resize 224 --gpus 7 --batch_size 32 --pretrained_name noisy-student
```

ii) Train a static heatmaps model (Figure 15 in paper), save results in folder `results`
```bash
python3 main_Unet.py --model_type unet --dropout 0.5 --epochs 35 --gamma 0.41731 --lr 0.0090552 --model_teacher timm-efficientnet-b0 --step_size 2 --scheduler --resize 224 --gpus 6 --batch_size 32 --pretrained_name noisy-student
```

### Disclaimer on reproducibility
We have done our best to ensure reproducibility of our results, however this is not always [guaranteed](https://pytorch.org/docs/stable/notes/randomness.html).
The output for some of our experiments as well as ranges for the hyper-parameter tuning can be found in the `resources` subfolder.
