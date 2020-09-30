import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger('eyegaze')
#Model edited from https://www.kaggle.com/salvation23/xray-cnn-pytorch/


class XRayNet(nn.Module):
    def __init__(self, hidden_dim=[512, 256], out_dim=64, dropout=0.3):
        super(XRayNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.batch1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.dense_layers = []
        self.out_dim = out_dim  # embedding dimensions, i.e. output size
        if len(hidden_dim) == 0:  # straight to final dense
            self.dropout1 = nn.Dropout(dropout, inplace=True)
            self.final_dense = nn.Linear(64 * 56 * 56, self.out_dim)
        else:
            for count, dim in enumerate(hidden_dim):
                insize = 64 * 56 * 56 if count == 0 else hidden_dim[count - 1]
                self.dense_layers.append(nn.Linear(insize, dim))
                self.dense_layers.append(nn.BatchNorm1d(dim, momentum=0.01))
            self.dense_layers = nn.ModuleList(self.dense_layers)
            self.dropout1 = nn.Dropout(dropout, inplace=True)
            self.final_dense = nn.Linear(hidden_dim[-1], self.out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.pool(self.batch1(x))
        x = x.view(-1, 64 * 56 * 56)
        for layer in self.dense_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.dropout1(x)
        x = self.final_dense(x) #(batch_size x emb_dim)
        return x