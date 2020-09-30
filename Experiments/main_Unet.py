import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from ray import tune
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from imgaug import augmenters as iaa
from datetime import datetime
from models.classifier import UnetClassifier
from utils.gradcam_utils import GradCam, visualize_gcam
from utils.dataset import split_dataset, EyegazeDataset
from utils.utils import cyclical_lr
from utils.dice_loss import GeneralizedDiceLoss
from utils.visualization import plot_roc_curve
from sklearn.metrics import roc_auc_score, roc_curve
plt.rcParams['figure.figsize'] = [25, 10]

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('unet')
logging.getLogger('matplotlib.font_manager').disabled = True
pil_logger = logging.getLogger('PIL').setLevel(logging.INFO)


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch Eye Gaze UNet')

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

    ## UNET Specific arguments.
    parser.add_argument('--model_type', default='unet', choices=['baseline', 'unet'], help='baseline, unet')
    parser.add_argument('--heatmaps_threshold', type=float, default=None, help='set the threshold value for the heatmap to be used with unet.')
    parser.add_argument('--gamma', type=float, default=1.0, help='Used to set the weighting value between the classifier and the segmentation in Unet')
    parser.add_argument('--model_teacher', type=str, default='timm-efficientnet-b0', help='model_teacher')
    parser.add_argument('--pretrained_name', type=str, default='noisy-student', help='model pretrained value')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--second_loss', type=str, default='ce', choices=['dice', 'ce'], help='Segmentation loss')

    # Misc
    parser.add_argument('--gpus', type=str, default='7', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--viz', default=False, action='store_true', help='[USE] Vizdom')
    parser.add_argument('--gcam_viz', default=False, action='store_true', help='[USE] Used for displaying the GradCam results')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test [same as train if not set]')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    return parser


def load_data(model_type, data_path, image_path, heatmaps_path, input_size, class_names, batch_size, num_workers, rseed, heatmaps_threshold):
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_file, valid_file, test_file = split_dataset(data_path, random_state=rseed)
    seq = iaa.Sequential([iaa.Resize((input_size, input_size))])
    image_transform = transforms.Compose([seq.augment_image, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    heatmap_static_transform = transforms.Compose([transforms.Resize([input_size, input_size]),
                                                   transforms.Grayscale(num_output_channels=1),
                                                   transforms.ToTensor()])
    static_heatmap_path = heatmaps_path
    train_dataset = EyegazeDataset(train_file, image_path, class_names, static_heatmap_path=static_heatmap_path,
                                   heatmaps_threshold=heatmaps_threshold, heatmap_static_transform=heatmap_static_transform,
                                   image_transform=image_transform)
    valid_dataset = EyegazeDataset(valid_file, image_path, class_names, static_heatmap_path=static_heatmap_path,
                                   heatmaps_threshold = heatmaps_threshold, heatmap_static_transform=heatmap_static_transform,
                                   image_transform=image_transform)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)

    test_dataset = EyegazeDataset(test_file, image_path, class_names, static_heatmap_path=static_heatmap_path,
                                  heatmaps_threshold = heatmaps_threshold, heatmap_static_transform=heatmap_static_transform,
                                  image_transform=image_transform)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=32)

    return train_dl, valid_dl, test_dl


def eval_net(model, loader, classifier_criterion, segment_criterion, model_type, gamma=0.5):
    "evaluation without the densecrf with the dice coefficient"
    model.eval()
    tot, seg, clas = 0, 0, 0
    counter = 0
    for images, labels, idx, X_hm, y_hm in (loader):
        images = images.cuda()
        labels = labels.cuda()
        if not model_type == 'baseline': y_hm = y_hm.cuda()
        l_segment = 0.
        with torch.no_grad():
            masks_pred, y_pred = model(images)
            if not model_type == 'baseline': l_segment = segment_criterion(masks_pred, y_hm)
            l_classifier = classifier_criterion(y_pred, labels)
            tot += ((gamma * l_classifier) + ((1 - gamma) * l_segment)).item()
            seg += l_segment
            clas += l_classifier
            counter += 1
    model.train()
    return tot/counter, seg/counter, clas/counter


def train_unet(args, model, train_dl, valid_dl, output_model_path, comment, tuning=False):
    if args.viz: writer = SummaryWriter(comment=comment)
    global_step = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])
    classifier_criterion = nn.BCEWithLogitsLoss()
    segment_criterion = GeneralizedDiceLoss() if args.second_loss == 'dice' else nn.BCEWithLogitsLoss() #nn.BCELoss() #

    for epoch in range(args.epochs):
        model.train()
        counter = 0
        epoch_loss = 0
        for images, labels, idx, X_hm, y_hm in (train_dl):
            images = images.cuda()
            labels = labels.cuda()
            if not args.model_type == 'baseline':
                y_hm = y_hm.cuda()

            masks_pred, y_pred = model(images)
            if not args.model_type == 'baseline':
                loss_segment = segment_criterion(masks_pred, y_hm)
            else:
                loss_segment = 0

            loss_classifier = classifier_criterion(y_pred, labels)
            total_loss = (args.gamma * loss_classifier) + ((1-args.gamma) * loss_segment)
            epoch_loss += total_loss.item()

            # logger.info(f"Classifier_Loss: {loss_classifier.item()}, Mask_Loss: {loss_segment.item()}")

            if args.viz:
                writer.add_scalar('Classifier_Loss', loss_classifier.item(), global_step)
                writer.add_scalar('Loss/Train', total_loss.item(), global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                if not args.model_type == 'baseline': writer.add_scalar('Mask_Loss', loss_segment.item(), global_step)

            optimizer.zero_grad()
            total_loss.backward()
            # -- clip the gradient at a specified value in the range of [-clip, clip].
            # -- This is mainly used to prevent exploding or vanishing gradients in the network training
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            global_step += 1
            counter += 1

        with torch.no_grad():
            val_loss, val_segment, val_classifier = eval_net(model, valid_dl, classifier_criterion, segment_criterion, args.model_type, args.gamma)
            if tuning:
                auc_scores = eval_eyegaze_network(model, valid_dl, args.class_names, '', 'debug', args.model_type, plot_data=False)
                tune.report(mean_accuracy=auc_scores)
                logging.info(f'Validation_Auc_scores: {auc_scores}')

        logging.info(f'Validation_Loss: {val_loss}')
        if not args.model_type == 'baseline': logging.info(f'Validation_Mask_Loss: {val_segment.item()}')
        logging.info(f"Validation Classifier loss: {val_classifier.item()}")
        if args.viz:
            writer.add_scalar('Validation_Loss', val_loss, global_step)
            if not args.model_type == 'baseline': writer.add_scalar('Validation_Mask_Loss', val_segment.item(), global_step)
            writer.add_scalar('Validation_Classifier_Loss', val_classifier.item(), global_step)
            writer.add_images('images', images, global_step)
            if args.model_type == 'unet':
                writer.add_images('masks/true', y_hm, global_step)
                writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        model.train()
        if args.scheduler: scheduler.step()
        try:
            os.makedirs(output_model_path)
            logger.info('Created Checkpoint directory')
        except OSError:
            pass
        torch.save(model.state_dict(), output_model_path + f"/Epoch_{epoch+1}.pth")
        logger.info(f"Checkpoint {epoch + 1} saved !")
    if args.viz: writer.close()


def eval_eyegaze_network(model, test_dl, class_names, model_dir, model_name, model_type, plot_data=True):
    model.eval()
    y = []
    y_hat = []
    # --- Prepare lists of predictions and ground truth
    for i in range(len(class_names)):
        y.append([])
        y_hat.append([])
    # --- Iterations
    with torch.no_grad():
        for images, y_batch, idx, X_hm, y_hm in (test_dl):
            images = images.cuda()
            y_batch = y_batch.cuda()
            masks_pred, output = model(images)
            output = nn.Sigmoid()(output)
            y_hat_batch = output.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            y_hat_batch = [np.array(y) for y in np.array(y_hat_batch).T.tolist()]
            y_batch = y_batch.transpose()
            for c, y_class in enumerate(y_batch):
                y[c].append(y_class)
            for c, y_class in enumerate(y_hat_batch):
                y_hat[c].append(y_class)
        y = [np.concatenate(c, axis=0) for c in y]
        y_hat = [np.concatenate(c, axis=0) for c in y_hat]
    logger.info('--' * 60)

    write_dir = os.path.join(model_dir, 'plots')
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)
    test_log_path = os.path.join(write_dir, os.path.splitext(model_name)[0] + ".log")
    logger.info(f"** write log to {test_log_path} **")
    aurocs = []
    fpr = dict()
    tpr = dict()
    with open(test_log_path, "w") as f:
        for i in range(len(class_names)):
            try:
                score = roc_auc_score(y[i], y_hat[i])
                aurocs.append(score)
                fpr[i], tpr[i], thresholds = roc_curve(y[i], y_hat[i])
            except ValueError:
                score = 0
                aurocs.append(score)
            f.write(f"{class_names[i]}: {score}\n")
            logger.info(f"{class_names[i]}: {score}")
        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write(f"mean auroc: {mean_auroc}\n")
        logger.info(f"mean auroc: {mean_auroc}\n")
    # Plot all ROC curves.
    if plot_data:
        logger.info("** plot and save ROC curves **")
        name_variable = os.path.splitext(model_name)[0] + ".png"
        plot_roc_curve(tpr, fpr, class_names, aurocs, write_dir, name_variable)
    return mean_auroc


def display_gcam(args, model_dir, model_name):
    if args.model_type == 'baseline':
        model = UnetClassifier(encoder_name=args.model_teacher, classes=n_segments,
                               encoder_weights=args.pretrained_name,
                               aux_params=aux_params).to(device=args.device)
    else:
        model = smp.Unet(args.model_teacher, classes=n_segments, encoder_weights=args.pretrained_name,
                         aux_params=aux_params).to(device=args.device)

    output_weights_name = os.path.join(model_dir, model_name)
    logger.debug(f'Loading Model: {output_weights_name}')
    if len(args.gpus.split(',')) > 1:
        print(f"Using {len(args.gpus.split(','))} GPUs!")
        device_ids = [int(i) for i in args.gpus.split(',')]
        model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(output_weights_name))

    # -- NOTE: MAJOR ASSUMPTION that both the baseline and the unet have the efficientnet-b0 as the encoder.
    # -- You will have to modify the candidate layers and the target layer accordingly if you change the model.
    candidate_layers = [f'encoder.blocks.{i}' for i in range(0, 7)]
    gcam = GradCam(model=model, candidate_layers=candidate_layers)
    logging.info(f"NOTE: This Gradcam is done for a specific target layer of the efficient-b0 on which the code was tested.")
    visualize_gcam(args, model, test_dl, gcam, target_layer='encoder.blocks.6', model_dir=model_dir)


if __name__ == '__main__':
    args = make_parser().parse_args()
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)
    cuda = torch.cuda.is_available() and args.gpus != '-1'
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    if cuda:
        torch.cuda.manual_seed(args.rseed)
        torch.cuda.manual_seed_all(args.rseed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.cuda.set_device("cuda:"+ args.gpus)
    args.device = torch.device("cuda:"+ args.gpus) if cuda else torch.device('cpu')
    logger.info(torch.cuda.get_device_name(args.device))
    if args.second_loss == 'ce': args.heatmaps_threshold = None

    # Create saving dir, all useful variables
    comment_variable = ''
    timestamp = str(datetime.now()).replace(" ", "").split('.')[0]
    for arg in vars(args):
        if arg not in ['data_path', 'heatmaps_path', 'image_path', 'class_names', 'gpus', 'viz', 'device',
                       'test', 'pretrained_name', 'testdir', 'output_dir', 'model_teacher', 'num_workers', 'rseed',
                       'pretrained']:
            comment_variable += f'{arg}{str(getattr(args, arg)).replace(" ", "")}_' \
                if arg != 'model_type' else f'{str(getattr(args, arg))}_'
    comment_variable += f'{timestamp}'

    logger.info(f"Comment Variable: {comment_variable}")
    output_model_path = os.path.join(args.output_dir, comment_variable)
    logger.info(f"[Arguments]:{args}")

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

    if not args.test:  # training
        if args.model_type == 'baseline':
            model = UnetClassifier(encoder_name=args.model_teacher, classes=n_segments, encoder_weights=args.pretrained_name,
                                   aux_params=aux_params).to(device=args.device)
        else:
            model = smp.Unet(args.model_teacher, classes=n_segments, encoder_weights=args.pretrained_name,
                           aux_params=aux_params).to(device=args.device)

        total_params_net = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        logger.info(f'Number of parameters: {total_params_net} ')
        if len(args.gpus.split(',')) > 1:
            print(f"Using {len(args.gpus.split(','))} GPUs!")
            device_ids = [int(i) for i in args.gpus.split(',')]
            model = nn.DataParallel(model, device_ids=device_ids)
        logger.info(f"Comment:{comment_variable}")
        train_unet(args, model, train_dl, valid_dl, output_model_path, comment_variable)

    logger.info('-- TESTING THE NETWORK OUTPUT --')
    best_mean_auc = 0.0
    best_model_name = ''
    model_dir = args.testdir if args.testdir else output_model_path
    for i in range(0, args.epochs, 1):
        model_name = f'Epoch_{i+1}.pth'
        if args.model_type == 'baseline':
            model = UnetClassifier(encoder_name=args.model_teacher, classes=n_segments,
                                   encoder_weights=args.pretrained_name,
                                   aux_params=aux_params).to(device=args.device)
        else:
            model = smp.Unet(args.model_teacher, classes=n_segments, encoder_weights=args.pretrained_name,
                           aux_params=aux_params).to(device=args.device)
        output_weights_name = os.path.join(model_dir, model_name)
        logger.info(f'Loading Model: {output_weights_name}')
        if len(args.gpus.split(',')) > 1:
            print(f"Using {len(args.gpus.split(','))} GPUs!")
            device_ids = [int(i) for i in args.gpus.split(',')]
            model = nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(torch.load(output_weights_name))
        model_auc = eval_eyegaze_network(model, test_dl, args.class_names, model_dir, model_name, args.model_type, plot_data=True)
        if model_auc >= best_mean_auc:
            best_model_name = model_name
            best_mean_auc = model_auc
            best_model = model
    logger.info(f"Best AUC:{best_mean_auc} from model with name: {best_model_name}")
    logger.info(f"Visualize the GRAD CAM for the best performing network.")
    if args.gcam_viz:
        display_gcam(args, model_dir, best_model_name)
