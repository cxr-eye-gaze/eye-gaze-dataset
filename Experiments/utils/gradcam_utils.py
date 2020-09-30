import os
import torch
import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F
from textwrap import wrap
from tqdm import tqdm as tqdm_write
from collections import OrderedDict

torch.set_printoptions(4, profile='short')


# -- Code modified from source: https://github.com/kazuto1011/grad-cam-pytorch
class _BaseWrapper(object):
    """
    Please modify forward() and backward() depending on your task.
    """
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def generate(self):
        raise NotImplementedError

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return list(zip(*self.probs.sort(0, True)))  # element: (probability, index)


class GradCam(_BaseWrapper):
    def __init__(self, model, candidate_layers=[]):
        super(GradCam, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers

        def forward_hook(module, input, output):
            self.fmap_pool[id(module)] = output.detach()


        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            if len(self.candidate_layers) == 0 or module[0] in self.candidate_layers:
                self.handlers.append(module[1].register_forward_hook(forward_hook))
                self.handlers.append(module[1].register_backward_hook(backward_hook))

    def find(self, pool, target_layer):
        # --- Query the right layer and return it's value.
        for key, value in pool.items():
            for module in self.model.named_modules():
                # print(module[0], id(module[1]), key)
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError(f"Invalid Layer Name: {target_layer}")

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads ,2))) + 1e-5
        return grads /l2_norm

    def compute_grad_weights(self, grads):
        grads = self.normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)


    def generate(self, target_layer):
        fmaps = self.find(self.fmap_pool, target_layer)
        grads = self.find(self.grad_pool, target_layer)
        weights = self.compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.0)

        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam


def compute_gradCAM(probs, labels, gcam, testing_labels, criterion, target_layer='encoder.blocks.6'):
    # --- one hot encode this:
    # one_hot = torch.zeros((labels.shape[0], labels.shape[1])).float()
    one_hot = torch.zeros((probs.shape[0], probs.shape[1])).float()
    max_int = torch.max(criterion(probs), 1)[1]

    if testing_labels:
        for i in range(one_hot.shape[0]):
            one_hot[i][max_int[i]] = 1.0

    else:
        for i in range(one_hot.shape[0]):
            one_hot[i][torch.max(labels, 1)[1][i]] = 1.0

    probs.backward(gradient=one_hot.cuda(), retain_graph=True)
    fmaps = gcam.find(gcam.fmap_pool, target_layer)
    grads = gcam.find(gcam.grad_pool, target_layer)

    weights = F.adaptive_avg_pool2d(grads, 1)
    gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    gcam_out = F.relu(gcam)
    return probs, gcam_out, one_hot


def get_mask(gcam, criterion, sigma=.5, w=8):
    for i in range(gcam.shape[0]):
        temp_loc = -1
        if gcam[i][:].sum() != 0:
            gcam[i][:] = gcam[i][:]
        else:
            temp_loc = i

        if temp_loc != -1:
            tqdm_write.write(f'#--Zero SUM Error{i}--#' * 2)

    gcam = F.interpolate(gcam, size=(224,224), mode='bilinear', align_corners=False)
    B, C, H, W = gcam.shape
    gcam = gcam.view(B, -1)
    gcam -= gcam.min(dim=1, keepdim=True)[0]
    gcam /= gcam.max(dim=1, keepdim=True)[0]
    mask = gcam.view(B, C, H, W)

    return mask


def wrap_plotting(args, images, gcam_mask, masks, indices, labels, model_dir, counter, y_cl, masks_pred=None):
    write_dir = os.path.join(model_dir, 'viz_plots')
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    for index in range(indices.shape[0]):
        raw_image = images[index].cpu().numpy()
        raw_image = np.rollaxis(raw_image, 0, 3)
        raw_image -= raw_image.min()
        raw_image /= raw_image.max()
        raw_image = np.multiply(raw_image, 255.0)

        # -- Plot ground truth masks.
        if args.model_type == 'unet':
            masks_pred = masks_pred.detach()
            unet_mask_pred_npy = masks_pred[index].cpu().numpy()
            unet_mask_pred_npy -= unet_mask_pred_npy.min()
            unet_mask_pred_npy /= unet_mask_pred_npy.max()
            unet_mask_pred_npy = np.multiply(unet_mask_pred_npy, 255.0)
            unet_mask_pred_npy = np.rollaxis(unet_mask_pred_npy, 0, 3)

        heatmap_image_npy = masks[index].cpu().numpy()
        heatmap_image_npy = np.multiply(heatmap_image_npy, 255.0)
        heatmap_image_npy = np.rollaxis(heatmap_image_npy, 0, 3)

        gcam_npy = gcam_mask[index].cpu().numpy()
        cmap = cm.jet_r(gcam_npy[0])[..., :3] * 255.0
        gcam_npy = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        gcam_npy[:, :, [2, 0]] = gcam_npy[:, :, [0, 2]]  # This is needed because opencv uses BGR as its default color while matplotlib uses RGB

        heatmap_image_npy = cv2.applyColorMap(np.uint8(heatmap_image_npy), cv2.COLORMAP_JET)
        heatmap_image_npy = heatmap_image_npy.astype(np.float) + raw_image.astype(np.float)
        heatmap_image_npy = heatmap_image_npy / heatmap_image_npy.max() * 255.0
        heatmap_image_npy[:, :, [2, 0]] = heatmap_image_npy[:, :, [0, 2]]

        if args.model_type == 'unet':
            unet_mask_pred_npy = cv2.applyColorMap(np.uint8(unet_mask_pred_npy), cv2.COLORMAP_JET)
            unet_mask_pred_npy = unet_mask_pred_npy.astype(np.float) + raw_image.astype(np.float)
            unet_mask_pred_npy = unet_mask_pred_npy / unet_mask_pred_npy.max() * 255.0
            unet_mask_pred_npy[:, :, [2, 0]] = unet_mask_pred_npy[:, :, [0, 2]]

        _, truth_labels = torch.max(labels[index], 0)
        label_name = args.class_names[truth_labels]
        _, pred_label = torch.max(y_cl[index], 0)
        pred_label_name = args.class_names[pred_label]


        if args.model_type == 'unet':
            fig, ax = plt.subplots(1, 4)
        else:
            fig, ax = plt.subplots(1, 3)

        plt.set_cmap('jet')
        plt.tight_layout()
        # -- Turn of the axis
        [axi.set_axis_off() for axi in ax.ravel()]
        # fig.tight_layout()
        plt0 = ax[0].imshow(np.uint8(raw_image), interpolation='bicubic')
        ax[0].set_title("\n".join(wrap(f'Index:{counter + index}, Truth: {label_name}')))

        plt1 = ax[1].imshow(np.uint8(gcam_npy), interpolation='bicubic')
        ax[1].set_title("\n".join(wrap(f'GradCam, Predicted: {pred_label_name}')))
        fig.colorbar(plt1, ax=ax[1], fraction=0.046, pad=0.04)

        plt2 = ax[2].imshow(np.uint8(heatmap_image_npy), interpolation='bicubic')
        ax[2].set_title("\n".join(wrap(f'EyeGaze Heatmap')))
        fig.colorbar(plt2, ax=ax[2], fraction=0.046, pad=0.04)

        if args.model_type == 'unet':
            plt3 = ax[3].imshow(np.uint8(unet_mask_pred_npy), interpolation='bicubic')
            ax[3].set_title("\n".join(wrap(f"Unet Prob Map")))
            fig.colorbar(plt3, ax=ax[3], fraction=0.046, pad=0.04)
            fig.savefig(f"{write_dir}/UNET_{counter}_{index}.png")
        else:
            fig.savefig(f"{write_dir}/BL_{counter}_{index}.png")

    return plt


def visualize_gcam(args, model, test_dl, gcam, target_layer = 'encoder.blocks.6', model_dir=''):
    counter = 0
    for images, labels, idx, X_hm, y_hm in tqdm_write(test_dl):
        images = images.cuda()
        labels = labels.cuda()
        y_hm = y_hm.cuda()
        prob_criterion = nn.Sigmoid()
        # --- When testing or debugging, turning this flag on will show the grad cam of the highest
        # --- output of the network instead of the truth label as in the training.
        testing_labels = True
        masks_pred, y_cl = model(images)
        _, gcam_out, one_hot = compute_gradCAM(y_cl, labels, gcam, testing_labels, prob_criterion, target_layer)
        gcam_mask = get_mask(gcam_out, prob_criterion)
        indices = one_hot.max(dim=1)[1]
        wrap_plotting(args, images, gcam_mask, y_hm, indices, labels, model_dir, counter, prob_criterion(y_cl), prob_criterion(masks_pred))
        counter += 1
