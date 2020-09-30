import os
import argparse
import torch
import sys
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from visdom import Visdom
from datetime import datetime
from torchvision import transforms
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader
from models.eyegaze_model import EyegazeModel
from utils.dataset import split_dataset, EyegazeDataset, collate_fn
from utils.utils import cyclical_lr, train_teacher_network, test_eyegaze_network, load_model

plt.rcParams['figure.figsize'] = [10, 10]

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger('eyegaze')
logging.getLogger('matplotlib.font_manager').disabled = True
pil_logger = logging.getLogger('PIL').setLevel(logging.INFO)


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')

    # Data
    parser.add_argument('--data_path', type=str, default='resources/master_sheet.csv', help='Data path')
    parser.add_argument('--image_path', type=str, default='/data/MIMIC/MIMIC-IV/cxr_v2/physionet.org/files/mimic-cxr/2.0.0', help='image_path')
    parser.add_argument('--heatmaps_path', type=str, help='Heatmaps directory',
                        default='/data/MIMIC/eye_gaze/fixation_heatmaps/uncalibrated/temporal_heatmaps')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--class_names', type=list, default=['Normal', 'CHF', 'pneumonia'], help='Label names for classification')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('--resize', type=int, default=224, help='Resizing images')

    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--scheduler', default=False, action='store_true', help='[USE] scheduler')
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')

    ## Temporal Model Specific arguments.
    parser.add_argument('--model_type', default='baseline', choices=['baseline', 'temporal'], help='model choice')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden size for image model')
    parser.add_argument('--emb_dim', type=int, default=64, help='cnn embedding size for heatmap model')
    parser.add_argument('--hidden_hm', nargs='+', type=int, default=[256, 128], help='hidden size for heatmap model')
    parser.add_argument('--num_layers_hm', type=int, default=1, help='num layers for heatmap model')
    parser.add_argument('--cell', type=str, default='lstm', choices=['lstm', 'gru'], help='LSTM or GRU for heatmap model')
    parser.add_argument('--brnn_hm', default=True, action='store_true', help='[USE] bidirectional for heatmap model')
    parser.add_argument('--attention', default=True, action='store_true', help='[USE] attention for heatmap model')

    # Misc
    parser.add_argument('--gpus', type=str, default='3', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--viz', default=False, action='store_true', help='[USE] Vizdom')
    parser.add_argument('--gcam_viz', default=False, action='store_true', help='[USE] Used for displaying the GradCam results')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test [same as train if not set]')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    return parser


def load_data(model_type, data_path, image_path, heatmaps_path, input_size, class_names, batch_size, num_workers, rseed):
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_file, valid_file, test_file = split_dataset(data_path, random_state=rseed)
    seq = iaa.Sequential([iaa.Resize((input_size, input_size))])
    image_transform = transforms.Compose([seq.augment_image, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if model_type in ['temporal']:
        heatmap_temporal_transform = transforms.Compose([transforms.Resize([input_size, input_size]),
                                                         transforms.Grayscale(num_output_channels=1),
                                                         transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                         transforms.Normalize(mean=mean, std=std)])
        heatmap_static_transform = transforms.Compose([transforms.Resize([input_size, input_size]), transforms.ToTensor()])
        static_heatmap_path = heatmaps_path
        train_dataset = EyegazeDataset(train_file, image_path, class_names, heatmaps_path=heatmaps_path,
                                       static_heatmap_path=static_heatmap_path,
                                       heatmap_temporal_transform=heatmap_temporal_transform,
                                       heatmap_static_transform=heatmap_static_transform,
                                       image_transform=image_transform)
        valid_dataset = EyegazeDataset(valid_file, image_path, class_names, heatmaps_path=heatmaps_path,
                                       static_heatmap_path=static_heatmap_path,
                                       heatmap_temporal_transform=heatmap_temporal_transform,
                                       heatmap_static_transform=heatmap_static_transform,
                                       image_transform=image_transform)
        test_dataset = EyegazeDataset(test_file, image_path, class_names, heatmaps_path=heatmaps_path,
                                      static_heatmap_path=static_heatmap_path,
                                      heatmap_temporal_transform=heatmap_temporal_transform,
                                      heatmap_static_transform=heatmap_static_transform,
                                      image_transform=image_transform)
        # drop_last=True for batchnorm issue: https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
        # this did not resolve the issue for all cases
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn, drop_last=True)
        valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn, drop_last=True)
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=32)
    else:
        train_dataset = EyegazeDataset(train_file, image_path, class_names, image_transform=image_transform)
        valid_dataset = EyegazeDataset(valid_file, image_path, class_names, image_transform=image_transform)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_dataset = EyegazeDataset(test_file, image_path, class_names, image_transform=image_transform)
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=32)
    return train_dl, valid_dl, test_dl


def run_experiment(args, train_dl, valid_dl, viz, env_name, output_model_path):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    image_classifier = EyegazeModel(args.model_type, len(args.class_names), dropout=args.dropout, emb_dim=args.emb_dim,
                                    hidden_dim=args.emb_dim, hidden_hm=args.hidden_hm, attention=args.attention,
                                    cell=args.cell, brnn_hm=args.brnn_hm, num_layers_hm=args.num_layers_hm).to(args.device)
    logger.info(image_classifier)
    total_params = sum([np.prod(p.size()) for p in image_classifier.parameters()])
    logger.info(f'Number of parameters:{total_params}')
    if len(args.gpus.split(',')) > 1:
        print(f"Using {len(args.gpus.split(',')) } GPUs!")
        device_ids = [int(i) for i in args.gpus.split(',')]
        image_classifier = nn.DataParallel(image_classifier, device_ids=device_ids)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(image_classifier.parameters(), lr=args.lr)
    clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
    exp_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])
    train_teacher_network(image_classifier, criterion, optimizer, exp_lr_scheduler, train_dl, valid_dl, output_model_path,
                          args.epochs, viz=viz, env_name=env_name,  is_schedule=args.scheduler)
    logger.info(f'Model saved at ...{output_model_path}')
    return image_classifier


if __name__ == '__main__':
    args = make_parser().parse_args()
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)
    cuda = torch.cuda.is_available() and args.gpus != '-1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cuda:
        torch.cuda.manual_seed(args.rseed)
        torch.cuda.manual_seed_all(args.rseed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.cuda.set_device("cuda:"+ args.gpus)
    args.device = torch.device("cuda:"+ args.gpus) if cuda else torch.device('cpu')
    logger.info(torch.cuda.get_device_name(args.device))
    # Create saving dir, all useful variables
    comment_variable = ''
    timestamp = str(datetime.now()).replace(" ", "").split('.')[0]
    for arg in vars(args):
        if arg not in ['data_path', 'heatmaps_path', 'image_path', 'class_names', 'gpus', 'viz', 'device', 'alpha', 'omega',
                        'lambda1', 'test', 'testdir', 'output_dir', 'model_teacher', 'num_workers', 'rseed', 'pretrained']:
            comment_variable += f'{arg}{str(getattr(args, arg)).replace(" ", "")}_' \
                if arg != 'model_type' else f'{str(getattr(args, arg))}_'
    comment_variable += f'{timestamp}'
    output_model_path = os.path.join(args.output_dir, comment_variable)
    logger.info("[Arguments]: %r", args)
    train_dl, valid_dl, test_dl = load_data(args.model_type, args.data_path, args.image_path, args.heatmaps_path,
                                            args.resize, args.class_names, args.batch_size, args.num_workers, args.rseed)
    if not args.test: #training
        viz = Visdom(env='EyeGaze', port=8097) if args.viz else None
        env_name = 'EyeGaze' if args.viz else None
        run_experiment(args, train_dl, valid_dl, viz, env_name=env_name, output_model_path=output_model_path)

    logger.info('---- NOW TESTING SET --- ')
    model_dir = args.testdir if args.testdir else output_model_path
    best_mean_auc = 0.0
    best_model_name = ''
    for i in range(0, args.epochs, 1):
        model_name = f'Epoch_{i}.pth'
        model = EyegazeModel(args.model_type, len(args.class_names), dropout=args.dropout,
                             emb_dim=args.emb_dim,
                             hidden_dim=args.emb_dim, hidden_hm=args.hidden_hm,
                             attention=args.attention,
                             cell=args.cell, brnn_hm=args.brnn_hm, num_layers_hm=args.num_layers_hm).to(args.device)
        if len(args.gpus.split(',')) > 1:
            print(f"Using {len(args.gpus.split(',')) } GPUs!")
            device_ids = [int(i) for i in args.gpus.split(',')]
            model = nn.DataParallel(model, device_ids=device_ids)
        model = load_model(model_name, model_dir, model).to(args.device)
        model_auc = test_eyegaze_network(model, test_dl, args.class_names, model_dir, model_name)
        if model_auc >= best_mean_auc:
            best_model_name = model_name
            best_mean_auc = model_auc
    logger.info(f"Best AUC:{best_mean_auc} from model with name: {best_model_name}")
