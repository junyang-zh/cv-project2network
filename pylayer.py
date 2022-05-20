import numpy as np
from math import *

'''
    LINEAR
    Implementation of the linear layer (also called fully connected layer)
    which performs linear transoformation on input data y = xW + b.
    This layer has two learnable parameters, weight of shape (input_channel, output_channel)
    and bias of shape (output_channel), which are specified and initalized in init_param()
    function. In this assignment, you need to implement both forward and backward computation
    Arguments:
        input_channel  -- integer, number of input channels
        output_channel -- integer, number of output channels
'''

class Linear(object):

    def __init__(self, input_channel, output_channel):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.init_param()

    def init_param(self):
        self.weight = (np.random.randn(self.input_channel,self.output_channel) * sqrt(2.0/(self.input_channel+self.output_channel))).astype(np.float32)
        self.bias = np.zeros((self.output_channel))

    '''
        Forward computation of linear layer, you may want to save some intermediate
        variable to class membership (self.) for reusing in backward computation.
        Arguments:
            input -- numpy array of shape (N, input_channel)

        Output:
            output -- numpy array of shape (N, output_channel)
    '''
    def forward(self, input):
        # to be compatible with conv2d
        self.input = input.reshape(input.shape[0], -1)
        ##################################################
        # TODO: YOUR CODE HERE: forward
        ##################################################
        # output = np.dot(self.input, self.weight)
        output = np.einsum('Ni,ij -> Nj', self.input, self.weight)
        return output

    '''
        Backward computation of linear layer, you need to compute the gradient
        w.r.t input, weight and bias respectively. You need to reuse the variable in forward
        computation to compute backward gradient.

        Arguments:
            grad_output -- numpy array of shape (N, output_channel)

        Output:
            grad_input -- numpy array of shape (N, input_channel), gradient w.r.t input
            grad_weight -- numpy array of shape (input_channel, output_channel), gradient w.r.t weight
            grad_bias --  numpy array of shape (output_channel), gradient w.r.t bias
    '''
    def backward(self, grad_input):
        ##################################################
        # TODO: YOUR CODE HERE: backward
        ##################################################        
        grad_bias = grad_input
        grad_weight = np.einsum('Ni,Nj -> ij', self.input, grad_input)
        grad_output = np.einsum('ij,Nj -> i', self.weight, grad_input)
        return grad_output, grad_weight, grad_bias

'''
    BatchNorm1D
    Implementation of batch normalization (or BN) layer, which performs normalization and rescaling
    on input data. Specifically, for input data X of shape (N,input_channel), BN layers firstly normalized the data along batch dimension
    by the mean E(x), variance Var(X) that are computed within batch data and both have shape of (input_channel)
    Then BN re-scales the normalized data with learnable parameters beta and gamma, both have shape of (input_channel).
    So the forward formula is written as

            Y = ((X - mean(X)) /  sqrt(Var(x) + eps)) * gamma + beta

    At the same time, BN layer maintains a running_mean and running_variance that are momentumly updated during
    forward iteration and would replace batch-wise E(x) and Var(x) for testing. The equations are:

            running_mean = (1 - momentum) * E(x)   +  momentum * running_mean
            running_var =  (1 - momentum) * Var(x) +  momentum * running_var

    During test time, since the batch size could be an arbitrary number, the statistic inside batch may not be a good approximation of data distribution,
    thus we need instead using running_mean and running_var to perform normalization.
    Thus the forward formular is modified to:

            Y = ((X - running_mean) /  sqrt(running_var + eps)) * gamma + beta

    Overall, BN maintains 4 learnable parameters with shape of (input_channel),
    running_mean, running_var, beta and gamma.  In this assignment, you need
    to complete the forward and backward computation and handle the case for

    Arguments:
        input_channel -- integer, number of input channel
        momentum      -- float,   the momentum value used for the running_mean and running_var computation
'''
class BatchNorm1d(object):

    def __init__(self, input_channel, momentum = 0.9):
        self.input_channel = input_channel
        self.momentum = momentum
        self.eps = 1e-3
        self.init_param()

    def init_param(self):
        self.r_mean = np.zeros((self.input_channel)).astype(np.float32)
        self.r_var = np.ones((self.input_channel)).astype(np.float32)
        self.beta = np.zeros((self.input_channel)).astype(np.float32)
        self.gamma = (np.random.rand(self.input_channel) * sqrt(2.0/(self.input_channel))).astype(np.float32)
    '''
        Forward computation of batch normalization layer and momentumly updated the running mean and running variance
        You may want to save some intermediate variables to class membership (self.) and you should take care of different behaviors
        during training and testing.

        Arguments:
            input -- numpy array (N, input_channel)
            train -- bool, boolean indicator to specify the running mode, True for training and False for testing
    '''
    def forward(self, input, train):
        ##########################################################################
        # TODO: YOUR CODE HERE
        ##########################################################################
        self.input = input
        if train:
            mu = np.mean(input, axis =0)
            var = np.var(input, axis =0)
            self.mu = mu
            self.var = var
            self.r_mean = self.r_mean * self.momentum + (1 - self.momentum) * mu
            self.r_var = self.r_var * self.momentum + (1 - self.momentum) * var
            self.input_norm = (input - mu[None,:]) / np.sqrt(var[None,:] + self.eps)
            output = (self.input_norm * self.gamma) + self.beta
        else:
            input_norm = (input - self.r_mean[None,:])/np.sqrt(self.r_var[None,:] + self.eps)
            output = (input_norm * self.gamma[None,:]) + self.beta[None,:]
        return output
    '''
        Backward computationg of batch normalization layer
        You need to write gradient w.r.t input data, gamma and beta
        It's recommend to follow the chain rule to firstly compute the gradient w.r.t to intermediate variable to
        simplify the computation.

        Arguments:
            grad_output -- numpy array of shape (N, input_channel)

        Output:
            grad_input -- numpy array of shape (N, input_channel), gradient w.r.t input
            grad_gamma -- numpy array of shape (input_channel), gradient w.r.t gamma
            grad_beta  -- numpy array of shape (input_channel), gradient w.r.t beta
    '''
    def backward(self, grad_output):
        ##########################################################################
        # TODO: YOUR CODE HERE
        ##########################################################################
        N = grad_output.shape[0]
        dxdhat = self.gamma[None,:] * grad_output
        output_term1 =  (1./N) * 1./np.sqrt(self.var + self.eps)
        output_term2 = N * dxdhat
        output_term3 = np.sum(dxdhat, axis=0)
        output_term4 = self.input_norm * np.sum(dxdhat * self.input_norm, axis=0)
        grad_input = output_term1 * (output_term2 - output_term3 - output_term4)
        grad_gamma = np.sum(grad_output * self.input_norm, axis = 0)
        grad_beta = np.sum(grad_output, axis = 0)
        return grad_input, grad_gamma, grad_beta

'''
    RELU
    Implementation of relu (rectified linear unit) layer. Relu is the no-linear activating function that
    set all negative values to zero and the formua is y = max(x,0).
    This layer has no learnable parameters and you need to implement both forward and backward computation
    Arguments:
        None
'''

class ReLU(object):
    def __init__(self):
        pass
    '''
        Forward computation of relu and you may want to save some intermediate variables to class membership (self.)
        Arguments:
            input -- numpy array of arbitrary shape

        Output:
            output -- numpy array having the same shape as input.
    '''
    def forward(self, input):
        self.input = input
        return np.maximum(input, 0)

    '''
        Backward computation of relu, you can either in-place modify the grad_output or create a copy.
        Arguments:
            grad_output-- numpy array having the same shape as input

        Output:
            grad_input -- numpy array has the same shape as grad_output. gradient w.r.t input
    '''
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_output[self.input<0] = 0
        return grad_input
'''
    CROSS_ENTROPY_LOSS_WITH_SOFTMAX
    Implementation of the combination of softmax function and cross entropy loss.
    In classification task, we usually firstly apply softmax to map class-wise prediciton
    into the probabiltiy distribution then we use cross entropy loss to maximise the likelihood
    of ground truth class's prediction. Since softmax includes exponential term and cross entropy includes
    log term, we can simplify the formula by combining these two functions togther so that log and exp term could cancell out
    mathmatically and we can avoid precision lost with float point numerical computation.
    If we ignore the index on batch sizel and assume there is only one grouth truth per sample,
    the formula for softmax and cross entropy loss are:
        Softmax: prob[i] = exp(x[i]) / \sum_{j}exp(x[j])
        Cross_entropy_loss:  - 1 * log(prob[gt_class])
    Combining these two function togther, we got
        cross_entropy_with_softmax: -x[gt_class] + log(\sum_{j}exp(x[j]))
    In this assignment, you will implement both forward and backward computation.
    Arguments:
        None
'''
class CrossEntropyLossWithSoftmax(object):
    def __init__(self):
        pass
    '''
        Forward computation of cross entropy with softmax, you may want to save some intermediate variables to class membership (self.)
        Arguments:
            input    -- numpy array of shape (N, C), the prediction for each class, where C is number of class
            gt_label -- numpy array of shape (N), it's a integer array and the value range from 0 to C-1 which
                        specify the ground truth class for each input
        Output:
            output   -- numpy array of shape (N), containing the cross entropy loss on each input
    '''
    def forward(self, input, gt_label):
        exp = np.exp(input)
        self.gt_label = gt_label
        self.prob = exp / np.sum(exp, axis = -1)[:,None]
        log_term = np.log(np.sum(exp, axis = -1))
        output = -input[np.arange(input.shape[0]), gt_label] + log_term
        return output

    '''
        Backward computation of cross entropy with softmax. It's recommended to resue the variable
        in forward computation to simplify the formula.
        Arguments:
            grad_output -- numpy array of shape (N)

        Output:
            output   -- numpy array of shape (N, C), the gradient w.r.t input of forward function
    '''
    def backward(self, grad_output):
        self.prob[np.arange(self.prob.shape[0]),self.gt_label] -= 1
        return grad_output[:,None] * self.prob

'''
    IM2COL
    For 4 dimensional input tensor with shape (N, C, H, W) where  N is the batch dimension,
    C is the channel dimension and H, W is the spatial dimension. The im2col functions flattens
    each slidding kernel-sized block (C * kernel_h * kernel_w) on each sptial location, so that
    the output has the shape of (N, (C * kernel_h * kernel_w), out_H, out_W) and we can thus formuate
    convolutional operation as matrix multiplication. The formula to
    compute out_H and out_W is same as to compute the output spatial size of convolutional layer.

    Arguments:
        input_data  -- numpy array of shape (N, C, H, W)
        kernel_h    -- integer, height of the sliding blocks
        kernel_w    -- integer, width of the sliding blocks
        stride      -- integer, stride of the sliding block in the spatial dimension
        padding     -- integer, zero padding on both size of inputs

    Returns:
        output_data -- numpy array of shape (N, (C * kernel_h * kernel_w), out_H, out_W)
'''

def im2col(input_data, kernel_h, kernel_w, stride, padding):
    ##########################################################################
    # todo: your code here
    ##########################################################################

    N, C, H, W = input_data.shape
    out_h = (H + 2*padding - kernel_h)//stride + 1
    out_w = (W + 2*padding - kernel_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))

    for y in range(kernel_h):
        y_max = y + stride*out_h
        for x in range(kernel_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.reshape(N, C * kernel_h * kernel_w, out_h, out_w)
    return col

'''
    COL2IM
    For a 4 dimensional input tensor with shape (N, (C * kernel_h * kernel_w), out_H, out_W)
    where  N is the batch dimension, C is the channel dimension and out_H, out_W is the spatial dimension,
    kernel_h and kernel_w are the specified kernel spatial dimension, col2im function calculated each combined value
    in the resulting array by summing all values from corresponding sliding kernel-sized block. With the same parameters,
    the output should has the same shape as input_data of im2col. This function serves as inverse subroutine of im2col and
    we can formuate the backward computation in convolutional layer as matrix multiplication

    Arguments:
        input_data  -- numpy array of shape (N, (C * kernel_H * kernel_W), out_H, out_W)
        kernel_h    -- integer, height of the sliding blocks
        kernel_w    -- integer, width of the sliding blocks
        stride      -- integer, stride of the sliding block in the spatial dimension
        padding     -- integer, zero padding on both size of inputs

    Returns:
        output_data -- output_array with shape (N, C, H, W)
'''

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):

    ##########################################################################
    # todo: your code here
    ##########################################################################
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, C, filter_h, filter_w, out_h, out_w)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

'''
   CONV2D
   Implementation of convolutional layer. This layer performs convolution between each sliding kernel-sized block
   and convolutional kernel. Unlike the convlution you implement in HW1 that you need flip the kernel,
   here the convolution operation could be simplified as cross-correlation (no need to flip the kernel).
   This layer has 2 learnable parameters, weight (convolutional kernel) and bias, which are specified and initalized
   in init_param() function. You need to complete both forward and backward function of the class. For backward, you need
   to compute the gradient w.r.t input, weight and bias respectively. The input argument: kernel_size, padding and stride jointly
   determine the output shape by the following formula

            out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1

   You need to use im2col, col2im inside forward and backward respectively, which formulates the
   sliding window scheme in convolutional layer as matrix multiplication.

   Arguments:
       input_channel  -- integer, number of input channel which should be the same as channel numbers of filter or input array
       output_channel -- integer, number of output channel produced by convolution or the number of filters
       kernel_size    -- integer or tuple, spatial size of convolution kernel. If it's tuple, it specifies the height and
                         width of kernel size.
       stride         -- integer, stride of convolution.
       padding        -- zero padding added on both sides of input array
'''

class Conv2d(object):
    def __init__(self, input_channel, output_channel, kernel_size, padding = 0, stride = 1):
        self.output_channel = output_channel
        self.input_channel = input_channel
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride
        self.init_param()

    def init_param(self):
        self.weight = (np.random.randn(self.output_channel, self.input_channel, self.kernel_h, self.kernel_w) * sqrt(2.0/(self.input_channel + self.output_channel))).astype(np.float32)
        self.bias = np.zeros(self.output_channel).astype(np.float32)

    '''
        Forward computation of convolutional layer, you need to use im2col in this function
        You may want to save some intermediate variables to class membership (self.)
        So that you can reuse this variable for backward computing
        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, output_chanel, out_H, out_W)
    '''
    def forward(self, input):
        ##########################################################################
        # TODO: YOUR CODE HERE
        ##########################################################################
        self.input = input
        N = input.shape[0]
        input_col = im2col(input, self.kernel_h, self.kernel_w, self.stride, self.padding)
        w_col = self.weight.reshape(self.output_channel,-1)
        col_output = (input_col[:,None,...] * w_col[None,:,:,None,None]).sum(axis = 2)
        self.input_col = input_col
        col_output += self.bias.reshape(1,-1,1,1)
        return col_output

    '''
        Backward computation of convolutional layer, you need col2im and saved variable from forward() in this function

        Arguments:
            grad_output -- numpy array of shape (N, output_channel, out_H, out_W)

        Ouput:
            grad_input  -- numpy array of shape(N, input_channel, H, W), gradient w.r.t input
            grad_weight -- numpy array of shape(output_channel, input_channel, kernel_h, kernel_w), gradient w.r.t weight
            grad_bias   -- numpy array of shape(output_channel), gradient w.r.t bias
    '''

    def backward(self, grad_output):
        ##########################################################################
        # TODO: YOUR CODE HERE
        ##########################################################################
        N= grad_output.shape[0]
        grad_b = np.sum(grad_output,axis = (0,2,3)).reshape(self.output_channel)
        grad_output_reshaped = grad_output.transpose(1,2,3,0).reshape(self.output_channel,-1)
        grad_w = (grad_output_reshaped[None,:,None,:] * self.input_col.reshape(N, 1, -1, grad_output_reshaped.shape[-1])).sum(axis = 0).sum(axis = -1)
        grad_w = grad_w.reshape(self.weight.shape)
        #grad_w = (np.dot(grad_output_reshaped,self.input_col)).sum(axis = 1).reshape(self.weight.shape)
        w_reshape = self.weight.reshape(self.output_channel,-1)
        grad_col = np.dot(w_reshape.T, grad_output_reshaped)
        grad_input = col2im(grad_col, self.input.shape, self.kernel_h, self.kernel_w, self.stride, self.padding)
        return grad_input,  grad_w, grad_b


'''
   MAXPOOL2D
   Implementation of max pooling layer. For each sliding kernel-sized block, maxpool2d compute the
   spatial maximum along each channels. This layer has no learnable parameter,
   You need to complete both forward and backward function of the layer. For backward, you need
   to compute the gradient w.r.t input. Similar as conv2d, the input argument, kernel_size, padding
   and stride jointly determine the output shape by the following formula

            out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1

   You need to use im2col, col2im inside forward and backward respectively. So that the computation
   could be simplified as taking max along certain direction.

   Arguments:
       kernel_size    -- integer or tuple, spatial size of convolution kernel. If it's tuple, it specifies the height and
                         width of kernel size.
       stride         -- integer, stride of convolution.
       padding        -- zero padding added on both sides of input array
'''
class MaxPool2d(object):
    def __init__(self, kernel_size, padding = 0, stride = 1):
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride

    '''
        Forward computation of max pooling layer, you need to use im2col in this function
        You may want to save some intermediate variables to class membership (self.) So that you
        can reuse this variable for backward computing
        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, input_channel, out_H, out_W)
    '''
    def forward(self, input):
        ##########################################################################
        # TODO: YOUR CODE HERE
        ##########################################################################
        self.input = input
        N, C, H, W = self.input.shape
        input_col = im2col(input, self.kernel_h, self.kernel_w, self.stride, self.padding)
        out_h, out_w = input_col.shape[-2:]
        input_col = input_col.reshape(N, C, -1,  out_h, out_w)
        self.argmax_coor = np.argmax(input_col, axis = 2)
        max_pool_feat = input_col.max(axis = 2)
        self.input_col = input_col
        # to be compatible with linear, save shape
        self.N, self.C, self.out_h, self.out_w = N, C, out_h, out_w
        return max_pool_feat

    '''
        Backward computation of max pooling layer,  you need col2im and saved variable
        from forward()

        Arguments:
            grad_output -- numpy array of shape (N, input_channel, out_H, out_W)

        Ouput:
            grad_input -- numpy array of shape(N, input_channel, H, W), gradient w.r.t input
    '''
    def backward(self, grad_output):
        # to be compatible with linear
        grad_output.reshape(self.N, self.C, self.out_h, self.out_w)
        ##########################################################################
        # TODO: YOUR CODE HERE
        ##########################################################################
        out_H, out_W =grad_output.shape[2:]
        N,C,in_H, in_W = self.input.shape
        grad_input = np.zeros_like(self.input_col).transpose(0,1,3,4,2)
        grad_input = grad_input.reshape(-1, grad_input.shape[-1])
        idx = np.arange(grad_input.shape[0])
        grad_input[idx, self.argmax_coor.reshape(-1)] = grad_output.reshape(-1)
        grad_input = grad_input.reshape(N,C,out_H,out_W, self.kernel_h, self.kernel_w).transpose(0,1,4,5,2,3)
        grad_input = col2im(grad_input, self.input.shape, self.kernel_h, self.kernel_w, stride = self.stride, pad = self.padding)
        return grad_input
