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
        inputs= ctx.saved_tensors
        d_input=histogram_layer.backward(inputs, d_histogram, ctx.bandwidth)
        return d_input

class Histogram(nn.Module):
    def __init__(self, region_size, bandwidth):
        super(Histogram, self).__init__()
        self.region_size=region_size
        self.bandwidth=bandwidth

    def forward(self, input):
        # get the size of the input
        N,SF,C,H,W=input.shape
        R=self.region_size
        # strides
        sN,sSF,sC,sH,sW=input.stride()
        # instead of iterating using for loops, we can factorize the computation by generating patches
        # overlapping patches of size (R x R)
        size=(N,SF,C,H-R+1,W-R+1,R,R)
        # strides
        stride=(sN,sSF,sC,sH,sW,sH,sW)
        # generate the patches
        # N x SF x C x H-R+1 x W-R+1 x R x R
        patches=input.as_strided(size, stride)
        # reshape to N*SF x C x (H-R+1) x (W-R+1) x R^2
        patches=patches.contiguous()
        patches=patches.view(N*SF,C,H-R+1,W-R+1,R*R)
        output=HistogramFunction.apply(patches, self.bandwidth)
        # add zero padding to the output to match the size of the input
        row_pad=torch.zeros(N*SF,C,H-R+1,int(R/2)).to(input.device)
        output=torch.cat([row_pad, output,row_pad], dim=3)
        col_pad=torch.zeros(N*SF,C,int(R/2),W).to(input.device)
        output=torch.cat([col_pad, output,col_pad], dim=2)
        output=output.view(N,SF,C,H,W)
        return output
