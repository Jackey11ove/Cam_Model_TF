from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
from torch.autograd import Function

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map."""
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def compute_priors(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)

        print(f"Prior boxes shape: {output.shape}")  # 调试信息
        return output


class PriorBoxFunction(Function):
    @staticmethod
    def forward(ctx, cfg):
        priorbox = PriorBox(cfg)
        output = priorbox.compute_priors()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None