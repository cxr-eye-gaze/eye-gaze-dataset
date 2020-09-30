import torch, logging
import torch.nn as nn
from .heatmaps_rnn import EncoderRNN
from .classifier import Classifier
from .simple_cnn import XRayNet

logger = logging.getLogger('eyegaze')


class EyegazeModel(nn.Module):
    def __init__(self, model_type, num_classes, dropout=0.5, num_layers_hm=3, cell='lstm',
                 brnn_hm=False, emb_dim=300, hidden_dim=64, hidden_hm=[512, 256], attention=True):
        super().__init__()
        self.model_type = model_type
        self.image_encoder = XRayNet(hidden_dim=hidden_hm, out_dim=hidden_dim, dropout=dropout)
        if self.model_type == 'temporal':
            self.heatmap_cnn = XRayNet(hidden_dim=hidden_hm, out_dim=hidden_dim, dropout=dropout)
            if len(hidden_hm) == 0:
                logger.info(f'Need at least one RNN hidden layer (hidden_hm), defaulting to {hidden_dim}...')
                hidden_lstm = hidden_dim
            else:
                hidden_lstm = hidden_hm[-1]
            self.heatmap_rnn = EncoderRNN(emb_dim, cell, brnn_hm, num_layers_hm, hidden_lstm, dropout, out_dim=hidden_dim, attention=attention)
        self.classifier = Classifier(num_classes, input_shape=hidden_dim * 2 if self.model_type == 'temporal' else hidden_dim)

    def forward(self, x):
        image, x_hm = x
        output = self.image_encoder(image)
        if self.model_type == 'temporal':
            output_heatmaps = self.heatmap_rnn(self.forward_heatmaps(x_hm))
            output = torch.cat([output, output_heatmaps], dim=1)
        output = self.classifier(output)
        return output

    def forward_heatmaps(self, x):  #(batch_size, #frames, 3, H, W)
        embed_seq = []
        for t in range(x.size(1)):
            x_out = self.heatmap_cnn(x[:, t, :, :, :])
            embed_seq.append(x_out)
        embed_seq = torch.stack(embed_seq, dim=0).transpose(0, 1)
        return embed_seq  #(batch_size, #frames, emb_dim)