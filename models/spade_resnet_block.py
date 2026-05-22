import torch
import torch.nn as nn
import torch.nn.functional as F
from .spade import SPADE


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # spade normalization layers
        self.spade_0 = SPADE(fin, label_nc)
        self.spade_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.spade_s = SPADE(fin, label_nc)

    def forward(self, x, segmap):
        x_s = self.shortcut(x, segmap)

        dx = self.spade_0(x, segmap)
        dx = F.relu(dx)
        dx = self.conv_0(dx)

        dx = self.spade_1(dx, segmap)
        dx = F.relu(dx)
        dx = self.conv_1(dx)

        out = x_s + dx
        return out

    def shortcut(self, x, segmap):
        if self.learned_shortcut:
            x_s = self.spade_s(x, segmap)
            x_s = self.conv_s(x_s)
        else:
            x_s = x
        return x_s
