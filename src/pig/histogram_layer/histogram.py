from torch import nn
from torch.autograd import Function
import torch

import histogram_layer

class HistogramFunction(Function):
    @staticmethod
    def forward(ctx, inputs, bandwidth):
        output=histogram_layer.forward(inputs, bandwidth)
        ctx.save_for_backward(inputs)
        ctx.bandwidth=bandwidth
        return output

    @staticmethod
    def backward(ctx, d_histogram):
        inputs= ctx.saved_tensors[0]
        d_input=histogram_layer.backward(inputs, d_histogram, ctx.bandwidth)
        # None for the gradient of the bandwidth
        return d_input , None

class Histogram(nn.Module):
    def __init__(self, bandwidth):
        super(Histogram, self).__init__()
        self.bandwidth=bandwidth

    def forward(self, input):
        # get the size of the input
        N,C,H,W=input.shape
        # reshape to N*SF*C x H x W
        input=input.contiguous()
        input=input.view(N*C,H,W)
        output=HistogramFunction.apply(input, self.bandwidth)
        # reshape the output to match the size of the input
        output=output.view(N,C,256)
        return output
