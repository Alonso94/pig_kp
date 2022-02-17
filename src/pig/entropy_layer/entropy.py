from torch import nn
from torch.autograd import Function
import torch

import entropy_layer

class EntropyFunction(Function):
    @staticmethod
    def forward(ctx, inputs, region_size, bandwidth):
        output=entropy_layer.forward(inputs, region_size, bandwidth)
        ctx.save_for_backward(inputs, region_size, bandwidth)
        return output

    @staticmethod
    def backward(ctx, d_entropy):
        inputs, region_size, bandwidth= ctx.saved_tensors
        d_input=entropy_layer.backward(inputs, d_entropy, region_size, bandwidth)
        return d_input

class Entropy(nn.Module):
    def __init__(self, region_size, bandwidth):
        super(Entropy, self).__init__()
        self.region_size=region_size
        self.bandwidth=bandwidth

    def forward(self, input):
        input=input.contiguous()
        return EntropyFunction.apply(input, self.region_size, self.bandwidth)

