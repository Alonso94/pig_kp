from torch import nn
from torch.autograd import Function
import torch

import joint_entropy

class JointEntropyFunction(Function):
    @staticmethod
    def forward(ctx, inputs, region_size, bandwidth):
        output=joint_entropy.forward(inputs, region_size, bandwidth)
        ctx.save_for_backward(inputs, region_size, bandwidth)
        return output

    @staticmethod
    def backward(ctx, d_entropy):
        inputs, region_size, bandwidth= ctx.saved_variables
        d_input=joint_entropy.backward(inputs, d_entropy, region_size, bandwidth)
        return d_input

class JointEntropy(nn.Module):
    def __init__(self, region_size, bandwidth):
        super(JointEntropy, self).__init__()
        self.region_size=region_size
        self.bandwidth=bandwidth

    def forward(self, input):
        input=input.contiguous()
        return JointEntropyFunction.apply(input, self.region_size, self.bandwidth)


    

