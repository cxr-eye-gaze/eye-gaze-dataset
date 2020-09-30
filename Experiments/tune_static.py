import torch
import sys
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import random
import numpy as np
import ray
import os
import logging
import torch.nn as nn
from main_Unet import load_data, train_unet
from models.classifier import UnetClassifier
import segmentation_models_pytorch as smp
from ray import tune
from ray.tune.schedulers import ASHAScheduler

DIR = 'tuneexps_unet'

# -- See eye-gaze-results.pptx for further tune settings and the top ROC values for both the unet and the baseline experiments.

def tune_eyegaze(config):
    args = type('', (), {})()
    args.model_type = 'baseline'#'unet'
    args.model_teacher = config['model_teacher'] #'timm-efficientnet-b0' #
    args.pretrained_name = 'noisy-student' if args.model_teacher == 'timm-efficientnet-b0' else 'imagenet'
    args.second_loss = 'ce'#config['second_loss'] #'dice' #

    args.lr = config["lr"]
    args.epochs = config["epochs"]
    args.heatmaps_threshold = None if args.second_loss == 'ce' else config['heatmaps_threshold']
    args.gamma = config['gamma']
    args.step_size = config['step_size']
    args.dropout = config['dropout']

    args.data_path = 'resources/master_sheet.csv'
    args.h5_path = '/data/MIMIC/images'
    args.heatmaps_path = '' # [Path to where the heatmap directory.]
    args.class_names = ['Normal', 'CHF', 'pneumonia']
    args.resize = 224
    args.num_workers = 16
    args.rseed = 42

    args.viz = False
    args.batch_size = 32
    args.gray_scale = True
    args.scheduler = True
    args.freeze = False
    args.output_dir = DIR

    # Create saving dir, all useful variables
    comment_variable = ''
    output_model_path = os.path.join(args.output_dir, comment_variable)

    if not os.path.exists(output_model_path): os.makedirs(output_model_path)
    logging.basicConfig(filename=os.path.join(output_model_path, 'log.log'), filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S', level=logging.DEBUG)
    logger = logging.getLogger('eyegaze')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False

    logging.getLogger('matplotlib.font_manager').disabled = True  # Matlab throws errors for fonts -.-
    logging.getLogger('PIL').setLevel(logging.INFO)

    logger.info("[Arguments]: "+' '.join([f'{arg}:{value}' for arg, value in sorted(vars(args).items())]))
    args.gpus = ",".join(map(str, ray.get_gpu_ids()))
    cuda = torch.cuda.is_available() and args.gpus != '-1'
    torch.cuda.set_device("cuda:0")
    args.device = torch.device("cuda") if cuda else torch.device('cpu')
    logger.info(torch.cuda.get_device_name(args.device))
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)
    torch.cuda.manual_seed(args.rseed)
    torch.cuda.manual_seed_all(args.rseed)

    train_dl, valid_dl, test_dl = load_data(args.model_type, args.data_path, args.image_path, args.heatmaps_path, args.resize,
                                            args.class_names, args.batch_size, args.num_workers, args.rseed, args.heatmaps_threshold)

    n_classes = len(args.class_names) # Classifier classes
    n_segments = 1 # Number of segmentation classes
    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=args.dropout,  # dropout ratio, default is None
        activation=None,  # activation function, default is None
        classes=n_classes,  # define number of output labels
    )

    if args.model_type == 'baseline':
        model = UnetClassifier(encoder_name=args.model_teacher, classes=n_segments,
                               encoder_weights=args.pretrained_name,
                               aux_params=aux_params).to(device=args.device)
    else:
        model = smp.Unet(args.model_teacher, classes=n_segments, encoder_weights=args.pretrained_name,
                         aux_params=aux_params).to(device=args.device)

    total_params_net = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logger.info(f'Number of parameters: {total_params_net} ')
    if len(args.gpus.split(',')) > 1:
        print(f"Using {len(args.gpus.split(','))} GPUs!")
        model = nn.DataParallel(model)
    train_unet(args, model, train_dl, valid_dl, output_model_path, comment_variable, tuning=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7' #'0,1,2,3'
ray.init(num_gpus=4)

search_space = {
    "epochs": tune.choice(list(range(20, 40, 5))),
    "lr": tune.uniform(1e-6,1e-2),
    "gamma": 1.0, #tune.uniform(0.0, 1.0),
    "step_size": tune.choice(list(range(2, 15, 3))),
    "dropout": tune.choice([0, 0.3, 0.5, 0.7]),
    "model_teacher": tune.choice(['densenet121', 'resnet101', 'timm-efficientnet-b0']),
    # "second_loss": tune.choice(['ce', 'dice']),
    # "heatmaps_threshold": tune.uniform(0, 1.0),#None, #
}

local_dir = './ray_tune_baseline_different_models'


analysis = tune.run(
    tune_eyegaze,
    resources_per_trial={'gpu': 1},
    num_samples=100,#40,
    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", grace_period=10, reduction_factor=3),
    config=search_space,
    local_dir=local_dir)

print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))
# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
df.to_csv(os.path.join(local_dir, 'tune_results.csv') ,index = False, header=True)
print(df)

