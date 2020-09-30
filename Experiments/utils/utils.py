import os
import logging
import math
import torch
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from .visualization import VisdomLinePlotter, plot_roc_curve
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

logger = logging.getLogger('eyegaze')

def cyclical_lr(step_sz=5, min_lr=0.001, max_lr=1, mode='triangular', scale_func=None, scale_md='cycles', gamma=1.):
# --- Sources:
# --- 1. https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20
# --- 2. https://www.jeremyjordan.me/nn-learning-rate/
# --- 3. https://github.com/bckenstler/CLR/blob/master/clr_callback.py
    if scale_func == None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycles'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1 / (2.**(x - 1))
            scale_mode = 'cycles'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma**(x)
            scale_mode = 'iterations'
        else:
            raise ValueError(f'The {mode} is not valid value!')
    else:
        scale_fn = scale_func
        scale_mode = scale_md

    lr_lambda = lambda iters: min_lr + (max_lr - min_lr) * rel_val(iters, step_sz, scale_mode)

    def rel_val(iteration, stepsize, mode):
        cycle = math.floor(1 + iteration / (2 * stepsize))
        x = abs(iteration / stepsize - 2 * cycle + 1)
        if mode == 'cycles':
            return max(0, (1 - x)) * scale_fn(cycle)
        elif mode == 'iterations':
            return max(0, (1 - x)) * scale_fn(iteration)
        else:
            raise ValueError(f'The {scale_mode} is not valid value!')
    return lr_lambda


# --- MAIN TRAINING Functions:
def train_teacher_network(model, criterion, optimizer, scheduler, train_dl, valid_dl, model_dir, num_epochs, viz=None, env_name=None, is_schedule=False):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    logger.info(f' MODEL FILE --> {model_dir}')
    logger.info(f'Scheduler == {is_schedule}')
    since = time.time()
    if viz:
        plotter = VisdomLinePlotter(viz, env_name=env_name)
        updatetextwindow = viz.text(f'Baseline loss metrics: {model_dir}')
        assert updatetextwindow is not None, 'Window was none'
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        model.train()
        counter = 0
        model_save_path = f'Epoch_{epoch}.pth'
        model_save_path = os.path.join(model_dir, model_save_path)
        if viz: plotter.plot('learning_rate', 'train', 'lr', counter + (epoch * len(train_dl)), scheduler.get_lr()[0])
        for images, labels, idx, X_hm, y_hm in tqdm(train_dl):
            if next(model.parameters()).is_cuda:
                images = images.cuda()
                labels = labels.cuda()
                if isinstance(X_hm, torch.Tensor):
                    X_hm = X_hm.cuda()
                if isinstance(y_hm, torch.Tensor):
                    y_hm = y_hm.cuda()
            # # zero the parameter gradients
            optimizer.zero_grad()
            x = images, X_hm
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            counter += 1
            if counter % 200 == 0:
                if viz:
                    plotter.plot('l_ce', 'train', 'Cross Entopy Loss - BL', counter + (epoch * len(train_dl)),
                         loss.cpu().detach().numpy())
                    viz.text('Eval Loss: {:.4f}'.format(loss.item()), win=updatetextwindow, append=True)
                tqdm.write(f'Eval Loss: {loss.item()}')
        if is_schedule: scheduler.step()
        model.eval()
        losses, nums, auc_scores = [], [], []
        with torch.no_grad():
            val_counter = 0
            for images, labels, idx, X_hm, y_hm in tqdm(valid_dl):
                if next(model.parameters()).is_cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                    if isinstance(X_hm, torch.Tensor):
                        X_hm = X_hm.cuda()
                    if isinstance(y_hm, torch.Tensor):
                        y_hm = y_hm.cuda()

                x = images, X_hm  # one input, to work with lr_finder
                y_t = model(x)
                # l_ce = F.cross_entropy(y_t, torch.max(labels, 1)[1])
                l_ce = criterion(y_t, labels)
                auc_scores.append(accuracy_score(y_t.argmax(dim=1).view(-1,1).cpu().numpy(), labels.argmax(dim=1).view(-1,1).cpu().numpy()))
                losses.append(l_ce)
                nums.append(len(images))
                val_counter += 1
        losses = np.asarray(losses)
        nums = np.asarray(nums)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        auc_score = np.sum(np.multiply(auc_scores, nums)) / np.sum(nums)
        if viz: viz.text('Val Loss: {:.4f}'.format(val_loss), win=updatetextwindow, append=True)
        logger.info('Val Loss: {:.4f}'.format(val_loss))
        torch.save(model.state_dict(), model_save_path)
        logger.info(f'SAVING MODEL FILE --> {model_save_path}')

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def load_model(model_name, model_dir, model):
    output_weights_name = os.path.join(model_dir, model_name)
    logger.info(f'MODEL FILE --- {output_weights_name}')
    if os.path.isfile(output_weights_name) == False:
        logger.info(f'No such model file: {output_weights_name}')
        return False
    model.load_state_dict(torch.load(output_weights_name))
    return model


def test_eyegaze_network(model, test_dl, class_names, model_dir, model_name):
    model.eval()
    y = []
    y_hat = []
    # --- Prepare lists of predictions and ground truth
    for i in range(len(class_names)):
        y.append([])
        y_hat.append([])
    # --- Iterations
    with torch.no_grad():
        for images, y_batch, idx, X_hm, y_hm in tqdm(test_dl):
            if next(model.parameters()).is_cuda:
                images = images.cuda()
                if isinstance(X_hm, torch.Tensor):
                    X_hm = X_hm.cuda()
            x = images, X_hm
            output = model(x)
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
    # Plot all ROC curves too.
    logger.info("** plot and save ROC curves **")
    name_variable = os.path.splitext(model_name)[0] + ".png"
    plot_roc_curve(tpr, fpr, class_names, aurocs, write_dir, name_variable)
    return mean_auroc