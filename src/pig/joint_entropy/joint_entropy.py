from torch import nn
from torch.autograd import Function
import torch

import joint_entropy_layer

class JointEntropyFunction(Function):
    @staticmethod
    def forward(ctx, inputs, bandwidth):
        output=joint_entropy_layer.forward(inputs, bandwidth)
        ctx.save_for_backward(inputs, bandwidth)
        ctx.bandwidth=bandwidth
        return output

    @staticmethod
    def backward(ctx, d_entropy):
        inputs= ctx.saved_variables
        d_input=joint_entropy_layer.backward(inputs, d_entropy, ctx.bandwidth)
        # None for the gradient of the bandwidth
        return d_input , None

class JointEntropy(nn.Module):
    def __init__(self, region_size, bandwidth):
        super(JointEntropy, self).__init__()
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
        # reshape to N x SF x C x (H-R+1) * (W-R+1) x R^2
        patches=patches.contiguous()
        patches=patches.view(N, SF, C,(H-R+1)*(W-R+1),R*R)
        # move channels to the last dimension
        patches=patches.permute(0,1,3,4,2)
        output=JointEntropyFunction.apply(patches, self.bandwidth)
        # reshape the output to N x SF x (H-R+1) x (W-R+1)
        output=output.view(N,SF,2,H-R+1,W-R+1)
        # add zero padding to the output to match the size of the input
        row_pad=torch.zeros(N,SF,2,H-R+1,int(R/2)).to(input.device)
        output=torch.cat([row_pad, output,row_pad], dim=4)
        col_pad=torch.zeros(N,SF,2,int(R/2),W).to(input.device)
        output=torch.cat([col_pad, output,col_pad], dim=3)
        output=output.view(N,SF,2,H,W)
        return output


    

