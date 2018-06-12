# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:40:20 2016

@author: Zhi Chen

Here we present a L-convolutional/pooling layer CNN model, 
i.e., trying to catpture the distinct features of the original images for detection, 
where again no ML libary is utilised for better understanding the whole procedure.
and only the numpy scientific computing module is employed for computation efficiency.
"""

import numpy as np
#import matplotlib.pyplot as plt
from scipy import linalg
#import numexpr as ne
import gc
#import os
#import sys
#import memory_profiler
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.cumath as cumath
import skcuda.linalg as culinalg
import skcuda.misc as cumisc
import timeit
culinalg.init()
"""
if python3:
"""
from six.moves import cPickle as cPickle
"""
elif: python2:
"""
#import cPickle

def _force_forder(x):
    """
    Converts arrays x to fortran order. Returns
    a tuple in the form (x, is_transposed).
    """
    if x.flags.c_contiguous:
        return (x.T, True)
    else:
        return (x, False)


"""
load data from files or url.
"""

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)#,encoding='iso-8859-1')
        return dict
    
def save_file(info, model_name):
    filename = model_name
    cPickle.dump(info, open(filename, 'wb'))
    return None

def Load_model(batchSize, model_name):
    model_data = unpickle(model_name)
    model = model_data['shadow_model']
    mask_shape = model_data['mask_shape']
    second_order = model_data['second_order']
    momentum = model_data['momentum']
    shadow_model = model_data['shadow_model']
    model['block_2']['0']['slope'] = np.asarray([[0.1,0.1],[0.1,0.1]])
    model['block_2']['1']['slope'] = np.asarray([[0.1,0.1],[0.1,0.1]])
    """
    To make sure that GReLU makes effect in the last block.                   
    """
    model['block_2']['0']['sec'] = np.asarray([[-34.0,-45.6],[-33.9,-32.0]])
    model['block_2']['1']['sec'] = np.asarray([[-38.0,-53.0],[-37.9,-36.0]])
    #second_order = Initial_momentum(model)
    #momentum = Initial_momentum(model)
    #shadow_model = Initial_momentum(model)
    L2 = 1e-8 * batchSize
    return model, L2, momentum,second_order,mask_shape,shadow_model
def Initial_momentum(model):
    momentum = dict()
    for key in model.keys():
        m_key = dict()
        if key=='out':
            """
            output layer model parameters.
            """
            for key2 in model[key]:
                m_key[key2]=np.zeros(model[key][key2].shape)
            momentum[key] = m_key
        else:
            """
            for each block model
            """
            for key2 in model[key].keys():
                """
                para of each layer
                """
                m_key2 = dict()
                for key3 in model[key][key2].keys():
                    """
                    inclusive of "sec" and "slope" keywords.
                    """
                    m_key2[key3]=np.zeros(model[key][key2][key3].shape)
                m_key[key2] = m_key2
            momentum[key] = m_key
    return momentum
    
def save_to_dict_m(model,g_model,alpha,mu):
    for key in model.keys():
        if key=='out':
            """
            softmax layer
            """
            for key2 in model[key].keys():
                model[key][key2] = mu * model[key][key2] + alpha * g_model[key][key2]
        else:
            """
            block para
            """
            for key2 in model[key].keys():
                for key3 in model[key][key2].keys():
                    model[key][key2][key3] = model[key][key2][key3] + alpha * g_model[key][key2][key3]
    return model

def save_to_dict_second_order(model,g_model,alpha,mu):
    for key in model.keys():
        if key=='out':
            """
            softmax layer
            """
            for key2 in model[key].keys():
                model[key][key2] = mu * model[key][key2] + alpha * g_model[key][key2]**2
        else:
            """
            block para
            """
            for key2 in model[key].keys():
                for key3 in model[key][key2].keys():
                    model[key][key2][key3] = model[key][key2][key3] + alpha * g_model[key][key2][key3]**2
    return model

def gradient_Relu(data):
    result = 1.0 * (data>=0)
    return result
    
def Relu(data):
    result = np.maximum(0,data)
    return result

def gradient_LRelu_para(diff,data,a):
    """
    data: input data before activation.
    diff: derivative of the activated output
    """
    batchSize = data.shape[0]
    gradient_a = -np.sum((data<0)*data*diff)/batchSize
    return gradient_a
    
def gradient_LRelu(data,a):
    """
    data is the input data before activation.
    """
    result = 1.0 * (data>=0) + a * (data<0)
    return result

def LRelu(data,a):
    result = np.maximum(0,data) + a * np.minimum(data,0)
    return result

def max_norm(weight,c):
    shape = weight.shape
    size = weight.size
    shape = list(shape)
    weight = weight.reshape([size//shape[-1],shape[-1]])
    weight_new = weight
    w_max_norm = np.sqrt(np.sum(weight**2,axis=0)).reshape([1,shape[-1]])
    for k in range(shape[-1]):
        if w_max_norm[0,k]>c:
            """
            constrain the weight vector on the surphace of a sphere for regularization.
            """
            L = w_max_norm[0,k]
            weight_new[:,k] = weight_new[:,k] * c / L
    weight_new = weight_new.reshape(tuple(shape))
    return weight_new
    
def permutation(data,label):
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    shuffled_label = label[permutation]
    return shuffled_data,shuffled_label

#@profile    
def ZCA_whitening(inputs, epsilon):
    cov = np.dot(inputs.T,inputs)/inputs.shape[0]
    U,s,V = np.linalg.svd(cov)
    temp = np.diag(1./np.sqrt(s+epsilon))
    gemm = linalg.get_blas_funcs("gemm",[temp,U.T])
    temp2 = gemm(alpha=1.0,a=temp,b=U.T,trans_a=False,trans_b=False)
    gemm2 = linalg.get_blas_funcs("gemm",[U,temp2])
    ZCAMatrix = gemm2(alpha=1.0,a=U,b=temp2,trans_a=False,trans_b=False)
    #ZCAMatrix = np.dot(U,temp2)
    del cov,U,s,V, temp,temp2
    return ZCAMatrix

def LoadData():
    trainData = np.empty( [0, 3072] )
    trainLabel = np.empty([0,1],dtype='int')
    for i in range(5):
        batch_i = unpickle('cifar-10-batches-py/data_batch_' + str(i+1))
        trainData = np.append(trainData, np.array(batch_i['data']), axis = 0)
        trainLabel = np.append(trainLabel, np.array(batch_i['labels'],dtype='int').reshape([10000,1]), axis = 0)
    test = unpickle('cifar-10-batches-py/test_batch')
    testData = np.array(test['data']).reshape([10000,3072])
    testLabel = np.array(test['labels'],dtype='int').reshape([10000,1])
    # a list containing label names
    dict_names = unpickle('cifar-10-batches-py/batches.meta')
    dict_names = dict_names['label_names']
    """
       global contrast normalization of pixel value.
       channel independent normalization
    """
    trainData = trainData.reshape([50000,3,1024])    
    trainMean = np.mean(trainData, axis=2).reshape([50000,3,1])
    trainStd = np.std(trainData, axis=2).reshape([50000,3,1])
    trainData = np.subtract(trainData, trainMean)
    trainData = np.divide(trainData, trainStd+1e-20)
    testData = testData.reshape([10000,3,1024])
    testMean = np.mean(testData,axis=2).reshape([10000,3,1])
    testStd = np.std(testData,axis=2).reshape([10000,3,1])
    testData = np.subtract(testData, testMean)
    testData = np.divide(testData, testStd+1e-20)
    trainData = trainData.reshape([50000,3072])
    testData = testData.reshape([10000,3072])    
    """
       ZCA whitening (per channel or as a whole).
    """
    ZCAMatrix = ZCA_whitening(trainData, 1e-2)
    trainMean = np.mean(trainData,axis=0).reshape([1,3072])
    trainData = np.dot(trainData-trainMean,ZCAMatrix.T)
    testData = np.dot(testData-trainMean,ZCAMatrix.T)
    trainData = trainData.reshape([trainData.shape[0],3,32,32])
    trainData = np.transpose(trainData,[0,2,3,1])
    testData = testData.reshape([testData.shape[0],3,32,32])
    testData = np.transpose(testData,[0,2,3,1])
    del ZCAMatrix
    gc.collect()
    return trainData,trainLabel,testData,testLabel,dict_names


def init_Grelu():
    """
    a small convnet with 3 conv layers and 
    a generalized Relu acitvation function 
    with (2*n+1) slopes.
    """
    depth = 1
    n = 2
    """
    initialization of parameters of the 
    activation functions.
    """
    sec = np.zeros((2*depth,n))
    slope = np.zeros((2*depth,n))
    for z in range(n):
        sec[1::2,z] = 0.2 + 0.2 * z
        sec[::2,z] =  -0.1 - 0.2 * z
        slope[1::2,z] = 1 + 5 * 1e-2 * (z+1)
        slope[::2,z] = 5 * 1e-2 * (z+1)
    return sec,slope


"""
Initialization of model parameters to be small random number array,
inclusive of weight matrix and the L2 regularization factor.
"""
def Initial(batchSize):
    """a feature map consists of 3*3 receptive field, resulting in 32*32 plane.
    total number of parameters: around 1M (for fair comparison with other models).
    64*3*9+64*64*2+64*64*9+64*64*2+64*64*9+64*64*1+64*64*1+480+640+640+640
    =0.574432M (but a version with around 1M is required for fair comparison)
    Employ padding on all feature maps before convolution to 
    safely maintain the edge information for detection.
    """
    numClass = 10
    #pad=True #default choice.
    model = dict()
    """
    cifar data size: 32*32*3
    """
    image_row = 32
    channel = 3
    mask_shape = dict()
    mask_shape['mask_v'] = (1,image_row,image_row,channel)
    patch_row = 3   
    #featureNum_in = 3  #number of RGB channels.
    image_size = image_row**2 * channel
    #featureNum = 1
    feature_list = np.array([3,128,160,192])
    numOfBlock = 3
    numOfLayer = 2
    out_in = image_size // 8**2
    mask_shape = dict()
    row = image_row
    for i in range(numOfBlock):
        feature_in = feature_list[i]
        feature_out = feature_list[i+1]
        block_i = dict()
        mask_shape['block_'+str(i)] = (1,row,row,feature_out)
        row = row // 2 #2*2 max-pooling is apllied after each block.
        for j in range(numOfLayer):
            layer_j = dict()
            sec,slope = init_Grelu()
            layer_j['sec'] = sec
            layer_j['slope'] = slope
            if j==0:
                f1 = feature_in
                f2 = feature_out
            else:
                f1 = feature_out
                f2 = f1
            if j==0:
                """
                inception of 1*1 at the last two layers of each block.
                """
                patch_row = 3
            else:
                """
                inception, bottleneck layer, but greatly reduce the number of parameters.
                """
                patch_row = 1
            layer_j['w_conv'] = 1. / np.sqrt((f1*patch_row**2)/2.) * (np.random.randn(patch_row**2,f1,f2))
            layer_j['b_conv'] = np.zeros([1,f2])
            layer_j['beta'] = 1. + 1 * 1e-2 * np.random.randn(1,f2)
            layer_j['gamma'] = 1e-2 * np.random.randn(1,f2)
            """
            each block stores layer para
            """
            block_i[str(j)] = layer_j
        """
        model stores block para.
        """
        if i<2:
            """
            keep 1 point per feature map for intermediate layers.
            """
            out_in += feature_out * 4
        else:
            """
            keep more info at the last convolved layer.
            i.e., only 2*2 avgPool is applied to get 
            16 features per feature map.
            """
            out_in += feature_out * 4
        model['block_'+str(i)] = block_i
    """
    max-pooling is employed and down-sample 
    the convolved feature maps 
    into smaller-size feature maps, typically 2*2.
    """
    """
    only 2*2 pooled feature from the raw image is used as residual information from lower layers.
    """
    print(model['block_0']['0']['slope'])
    softmax = dict()
    w_o = 1. / np.sqrt((out_in)/2.) * (np.random.randn(out_in,numClass))   
    b_o = 0 * (np.random.randn(1,numClass))
    beta_o = 1. + 1 * 1e-2 * np.random.randn(1,numClass)
    gamma_o = 1e-2 * np.random.randn(1,numClass)
    softmax['w_o'] = w_o
    softmax['b_o'] = b_o
    softmax['beta'] = beta_o
    softmax['gamma'] = gamma_o
    mask_shape['mask_res'] = (1,out_in)
    model['out'] = softmax
    print(model.keys())
    L2 = 1e-8 * batchSize  #as 400 is batchSize.
    momentum = Initial_momentum(model)
    second_order = Initial_momentum(model)
    """
    be cautious that copy dict() object is the same object, as they are in the same memory area. 
    """
    shadow_model = Initial_momentum(model)
    gc.collect()
    return model,L2,momentum,second_order,mask_shape,shadow_model

"""
Training process.
"""
"""
gradient check
"""
def gradientcheck(deri_num,derivative):
    if abs(deri_num-derivative[0,0]) / abs(deri_num) > 4 * 1e-2 or deri_num * derivative[0,0]<0:
    #if abs(deri_num-derivative[0]) / abs(deri_num) > 1e-3 or deri_num * derivative[0]<0:
        print('gradient check fails!')
    else:
        print('gradient check success!')
    print('deri_num:'+str(deri_num))
    print('derivative:'+str(derivative[0,0]))
    print('all:'+str(derivative))
    print('ratio:'+str(deri_num/derivative[0,0]))
    #print('derivative:'+str(derivative[0]))
    #pint('ratio:'+str(deri_num/derivative[0]))
    return None
  
##@profile
def trainBatch(step,alpha,batchData,batchLabel,model,shadow_model,L2,momentum,second_order,epoch,mask_shape):
    mask = dict()
    v_shape = batchData.shape
    v_shape = list(v_shape)
    v_shape[0] = 1
    v_shape = tuple(v_shape)
    mask['p_v'] = 0.1
    mask['p_fc'] = 0.5
    mask['mask_v'] = np.random.binomial(1,1-mask['p_v'],v_shape)
    for block in range(len(model)-1):
        shape_i = mask_shape['block_'+str(block)]
        mask_i = dict()
        mask_i['p_conv'] = 0.5
        """
        dropout only applies to the first conv layer of each block.
        """
        mask_i['mask'] = np.random.binomial(1,1-mask_i['p_conv'],shape_i)
        mask['block_'+str(block)] = mask_i
    """
    normRes is for a 2D tensor.
    """
    normRes = mask_shape['mask_res']
    mask['mask_res'] = np.random.binomial(1,1-mask['p_fc'],normRes)
    cache,loss,batchComp = batch(batchData,batchLabel,model,L2,mask,train=True)
    sign = 0
    if sign == 1:
        model_u = model.copy()
        epsilon = 1e-5
        b_u = model_u['block_2']['1']['sec']
        #print(b_u)
        #b_u[0,0,0] = b_u[0,0,0] + epsilon
        #b_u = model['block_2']['0']['beta']
        b_u[0,0] = b_u[0,0] + epsilon
        model_u['block_2']['1']['sec'] = b_u
        #model_u['out']['w_o'] = b_u
        #print(b_u)
        #print('model_diff:'+str(model_u['block_1']['0']['sec']-model['block_1']['0']['sec']))
        cache_2,loss_2,batchComp_2 = batch(batchData,batchLabel,model_u,L2,mask,train=True)
        #k = k-2
        #if checking dropout, should be with the same mask.
        #therefore, if can be provided before going to the batch function.
        #simple checking is OK.
        deri_num =  (loss_2-loss) / epsilon
        print('loss in trainBatch function:'+str(loss))
        print('loss in trainBatch function:'+str(loss_2))
    if np.isnan(loss) or not np.isfinite(loss):
        #in case something goes wrong.
        print('loss is not a number')
        return None
    #t0 = timeit.default_timer()
    derivative,model,momentum,second_order,shadow_model = descent(step,alpha,batchData,batchLabel,model,shadow_model,L2,momentum,second_order,cache,epoch,mask)
    #t1 = timeit.default_timer()
    #print('descent_time_consumption: ',t1-t0)
    if sign == 1:
        gradientcheck(deri_num,derivative)
    gc.collect()
    return model,batchComp,loss,momentum,second_order,shadow_model
"""
method used for updating parameters---backpropagation.
"""
"""
updating descent: pooling layer
"""
"""
utility function.
"""
def conv_pool_mapping(pool,scale):
    """assumping square shape of rows and columns for each feature.
    #2: downsample scale.
    """
    pool_row = pool.shape[1]
    conv = np.zeros([pool.shape[0],pool_row*scale,pool_row*scale,pool.shape[3]])
    for m in range(scale):
        for n in range(scale):
            conv[:,m::scale,n::scale,:] = pool
    return conv

def pool_descent(batchConv,pool_diff,batchPoolSign):
    """
    batchPoolSign and batchConv:batchSize*patchrow*patchrow*featureNum
    batchPool:batchSize*(patchrow/2)*(patchrow/2)**featureNum
    batchConvZ:retained valid batchConv maximum values, 
               with the sampe shape of batchPool
    pool_diff: 4D array
    retain only valid values, i.e., maximum retained in pooling batchConv.
    This function applies to any pooling layer.
    Further, in this program, we only consider 2*2 pooling/subsampling.
    """
    scale = 2
    pool_conv = conv_pool_mapping(pool_diff,scale)
    conv_diff = pool_conv * batchPoolSign
    
    
    """
    For sigmoid activation propagation, can be employed in api mode.
    conv_diff = conv_diff * batchConv * (1-batchConv)
    For tanh activation propagation, we have
    This was performed after the normalization layer
    """
    del pool_conv
    return conv_diff
"""
convolution propatation for all convolution layers.
Implementation on full-connected feature Map case.
"""
def conv_descent_3(data,batchConv,w_conv,b_conv,conv_diff,L2,pad=True):
    """
    w_conv:: (patchRow**2) * featureNum * featureNum2
    b_conv:: 1 * featureNum2
    batchLower:: batchSize * feature_row * feature_row * featureNum
    batchConv:: batchSize * conv_row * conv_row * featureNum2
    descent of convolution layer weight matrix and bias vector.
    conv_diff: shape of batchConv.
    This applies to all convolution layers where full connection 
    for all color channels is assumed.
    """
    lower_shape = data.shape
    batchSize = data.shape[0]
    featureNum = data.shape[3]
    featureNum2 = batchConv.shape[3]
    row = data.shape[1]
    patch_row = int(np.sqrt(w_conv.shape[0]))
    if pad==True and patch_row==3:
        batchLower = np.zeros((batchSize,row+2,row+2,featureNum))
        batchLower[:,1:-1,1:-1,:] = data
    else:
        batchLower = data
    conv_row = batchConv.shape[1]
    conv_map = parallel_conv_2(batchLower,patch_row)
    conv_map = conv_map.reshape([batchSize*conv_row**2,patch_row**2*featureNum])
    conv_map = np.transpose(conv_map)
    conv_diff_map = conv_diff
    conv_diff_map = conv_diff_map.reshape([batchSize*conv_row**2,featureNum2])
    gemm = linalg.get_blas_funcs(["gemm"],[conv_map,conv_diff_map])[0]
    temp = gemm(alpha=1.0,a=conv_map,b=conv_diff_map,trans_a=False,trans_b=False)
    w_derivative_conv = -temp.reshape([patch_row**2,featureNum,featureNum2]) + L2 * w_conv
    b_derivative_conv = -np.sum(conv_diff_map,0).reshape([featureNum2,1]) / batchSize
    w_derivative_conv = w_derivative_conv / batchSize
    w_conv_map = w_conv.reshape([patch_row**2*featureNum,featureNum2])
    w_conv_map = np.transpose(w_conv_map)
    gemm_d = linalg.get_blas_funcs(["gemm"],[conv_diff_map,w_conv_map])[0]
    lower_diff_map = gemm_d(alpha=1.0,a=conv_diff_map,b=w_conv_map,trans_a=False,trans_b=False)
    lower_diff_map = lower_diff_map.reshape([batchSize,conv_row,conv_row,patch_row**2,featureNum])
    lower_diff = np.zeros(batchLower.shape)
    """
    apply the difference evenly on each element of input.
    """
    if pad==True and patch_row==3:
        pad_row = row + 2
    else:
        pad_row = row
    for j in range(patch_row**2):
        k,l = j//patch_row,np.mod(j,patch_row)
        lower_diff[:,k:pad_row-patch_row+k+1,l:pad_row-patch_row+l+1,:] = lower_diff[:,k:pad_row-patch_row+k+1,l:pad_row-patch_row+l+1,:] + lower_diff_map[:,:,:,j,:].reshape([batchSize,conv_row,conv_row,featureNum])
    if pad==True and patch_row==3:
        lower_diff = lower_diff[:,1:-1,1:-1,:].reshape(lower_shape)
    del w_conv_map
    del gemm_d
    del lower_diff_map
    del batchLower
    del conv_map
    #del conv_diff_map
    return w_derivative_conv,b_derivative_conv,lower_diff

"""
gpu implementation of conv_descent_3
"""
def conv_descent_3_gpu(data,batchConv,w_conv,b_conv,conv_diff,L2,pad=True):
    """
    w_conv:: (patchRow**2) * featureNum * featureNum2
    b_conv:: 1 * featureNum2
    batchLower:: batchSize * feature_row * feature_row * featureNum
    batchConv:: batchSize * conv_row * conv_row * featureNum2
    descent of convolution layer weight matrix and bias vector.
    conv_diff: shape of batchConv.
    This applies to all convolution layers where full connection 
    for all color channels is assumed.
    Note that all inputs are numpy arrays in cpu.
    """
    lower_shape = data.shape
    batchSize = data.shape[0]
    featureNum = data.shape[3]
    featureNum2 = batchConv.shape[3]
    row = data.shape[1]
    patch_row = int(np.sqrt(w_conv.shape[0]))
    if pad==True and patch_row==3:
        batchLower_cpu = np.zeros((batchSize,row+2,row+2,featureNum))
        batchLower_cpu[:,1:-1,1:-1,:] = data
        #batchLower = gpuarray.to_gpu(batchLower_cpu)
    else:
        #batchLower = data
        batchLower_cpu = data
    conv_row = batchConv.shape[1]
    conv_map_cpu = parallel_conv_2(batchLower_cpu,patch_row)
    conv_map = gpuarray.to_gpu(conv_map_cpu)
    conv_map = conv_map.reshape([batchSize*conv_row**2,patch_row**2*featureNum])
    conv_map = culinalg.transpose(conv_map)
    conv_diff_map = gpuarray.to_gpu(conv_diff)
    conv_diff_map = conv_diff_map.reshape([batchSize*conv_row**2,featureNum2])
    w_derivative_conv = culinalg.dot(conv_map,conv_diff_map)
    w_derivative_conv = -w_derivative_conv.reshape([patch_row**2,featureNum,featureNum2])
    w_conv_gpu = gpuarray.to_gpu(w_conv)
    #b_conv_gpu = gpuarray.to_gpu(b_conv)
    w_derivative_conv = cumisc.add(w_derivative_conv,L2 * w_conv_gpu)
    b_derivative_conv = -cumisc.sum(conv_diff_map,axis=0).reshape([featureNum2,1]) / batchSize
    w_derivative_conv = w_derivative_conv / batchSize
    w_conv_map = w_conv_gpu.reshape([patch_row**2*featureNum,featureNum2])
    w_conv_map = culinalg.transpose(w_conv_map)
    lower_diff_map = culinalg.dot(conv_diff_map,w_conv_map)
    lower_diff_map = lower_diff_map.reshape([batchSize,conv_row,conv_row,patch_row**2,featureNum])
    del conv_map
    del conv_diff_map
    del conv_map_cpu
    del w_conv_gpu
    lower_diff_map_cpu = lower_diff_map.get()
    del lower_diff_map
    lower_diff = np.zeros(batchLower_cpu.shape)
    """
    apply the difference evenly on each element of input.
    """
    g_w_conv_cpu = w_derivative_conv.get()
    g_b_conv_cpu = b_derivative_conv.get()
    del w_derivative_conv
    del b_derivative_conv
    if pad==True and patch_row==3:
        pad_row = row + 2
    else:
        pad_row = row
    for j in range(patch_row**2):
        k,l = j//patch_row,np.mod(j,patch_row)
        lower_diff[:,k:pad_row-patch_row+k+1,l:pad_row-patch_row+l+1,:] = lower_diff[:,k:pad_row-patch_row+k+1,l:pad_row-patch_row+l+1,:] + lower_diff_map_cpu[:,:,:,j,:].reshape([batchSize,conv_row,conv_row,featureNum])
    if pad==True and patch_row==3:
        lower_diff = lower_diff[:,1:-1,1:-1,:].reshape(lower_shape)
        lower_diff = lower_diff.reshape(lower_shape)
    del lower_diff_map_cpu
    gc.collect()
    return g_w_conv_cpu,g_b_conv_cpu,lower_diff

def parallel_conv_2_gpu(batch_map,patchRow):
    #batch_map: dim:: batchSize * featureRow * featureRow * featureNum
    featureRow = batch_map.shape[1]
    featureNum = batch_map.shape[3]
    convRow = featureRow - patchRow + 1
    batchSize = batch_map.shape[0]
    conv_map = cumisc.zeros([batchSize,convRow**2,patchRow,patchRow,featureNum],batch_map.dtype)
    for k in range(convRow**2):
        m,n = k//convRow,np.mod(k,convRow)
        conv_map[:,k,:,:,:] = batch_map[:,m:m+patchRow,n:n+patchRow,:]
    conv_map = conv_map.reshape([batchSize,convRow**2,patchRow**2,featureNum])
    return conv_map

"""
backpropagation of normalization layer processing.
"""
def norm_descent(diff,batchLower,batchNorm,batchMean,batchStd,beta,gamma,L2):
    tiny = 1e-20
    shape = batchLower.shape
    batchSize = batchLower.shape[0]
    size = batchLower.size
    if len(shape)>2:
       batchSize2 = batchSize*shape[1]**2
    else:
       batchSize2 = batchSize
    paraNum = size // batchSize2
    batchNorm = batchNorm.reshape([batchSize2,paraNum])
    batchBar = np.divide(batchNorm-gamma,beta)
    diff = diff.reshape([batchSize2,paraNum])
    lower_diff = np.divide(diff,batchStd + tiny)
    lower_diff -= np.divide(np.mean(diff,axis=0),batchStd +tiny)
    lower_diff -= np.divide(np.mean(diff * batchBar,axis=0),batchStd +tiny) * batchBar
    lower_diff = lower_diff * beta
    lower_diff = lower_diff.reshape(shape)
    beta_derivative = -np.sum(diff * batchBar,axis=0)
    beta_derivative = beta_derivative.reshape([1,paraNum]) / batchSize
    gamma_derivative = -np.sum(diff,axis=0)
    gamma_derivative = gamma_derivative.reshape([1,paraNum]) / batchSize
    del batchBar
    return lower_diff,beta_derivative,gamma_derivative
"""
determining train rate based on correct rate.
momentum will be applied in this version.
"""    
def training_rate(epoch):
    mu = 0.9
    mv = 0.999
    """
    max-norm regularization.
    """
    c = 4
    return mu,mv,c

def adam(step,epoch,model,momentum,second_order,mu,mv,alpha,eps,c):
    for key in model.keys():
        if key=='out':
            model['out']['w_o'] -= alpha * (momentum['out']['w_o'] /(1-mu**step))/((np.sqrt(second_order['out']['w_o']/(1-mv**step)) + eps))
            model['out']['b_o'] -= alpha * (momentum['out']['b_o'] /(1-mu**step))/((np.sqrt(second_order['out']['b_o']/(1-mv**step)) + eps))
            model['out']['beta'] -= alpha * (momentum['out']['beta'] /(1-mu**step
))/((np.sqrt(second_order['out']['beta']/(1-mv**step)) + eps))
            model['out']['gamma'] -= alpha * (momentum['out']['gamma'] /(1-mu**step
))/((np.sqrt(second_order['out']['gamma']/(1-mv**step)) + eps))
            model['out']['w_o'] = max_norm(model['out']['w_o'],c)
        else:
            """
            block
            """
            for key2 in model[key].keys():
                for key3 in model[key][key2].keys():
                    if key3 == 'sec' or key3=='slope':
                        #print('freeze update until several epoches training has been performed.')
                        if epoch>=0:
                            """
                            freeze update until the model has reached some good starting point for GRelu update.
                            """
                            model[key][key2][key3] -=  min(1e-3,alpha) * (momentum[key][key2][key3]/(1-mu**step)) / ((np.sqrt(second_order[key][key2][key3] /(1-mv**step)) + eps))
                    elif key3 != 'b_conv':
                        """
                        b_conv is compensated by normalization layers. redundant design and should be deleted. 
                        """
                        model[key][key2][key3] -= alpha * (momentum[key][key2][key3]/(1-mu**step)) / ((np.sqrt(second_order[key][key2][key3] /(1-mv**step)) + eps))
                    if key3 == 'w_conv':
                        """
                        max norm.
                        """
                        #print('w_conv weight normalization! testing if it works!')
                        model[key][key2][key3]  = max_norm(model[key][key2][key3],c)
                    
    return model
    
#@profile
def descent(step,alpha,batchData,batchLabel,model,shadow_model,L2,momentum,second_order,cache,epoch,mask):
    pad = True
    mu,mv,c = training_rate(epoch)
    g_model = dict()
    batchData = batchData * mask['mask_v']/(1-mask['p_v'])
    batchSize = batchLabel.shape[0]
    batchProb = cache['out']['layer_out']
    numClass = batchProb.shape[1]
    batchLabelExpanded = np.zeros([batchSize,numClass])
    batchLabelExpanded[np.arange(batchSize), batchLabel.reshape([1,batchSize])] = 1
    diff = batchLabelExpanded - batchProb
    batchDiff,g_beta_o,g_gamma_o = norm_descent(diff,cache['out']['norm'],cache['out']['normed'],cache['out']['norm_mean'],cache['out']['norm_std'],model['out']['beta'],model['out']['gamma'],L2)
    """
    batchRes is a input of the descent function.
    """
    batchRes = cache['out']['layer_in']
    res_T = np.transpose(batchRes)
    w_o = model['out']['w_o']
    b_o = model['out']['b_o']
    gemm_o = linalg.get_blas_funcs(["gemm"],[res_T,batchDiff])[0]
    d_o = gemm_o(alpha=1.0,a=res_T,b=batchDiff,trans_a=False,trans_b=False) 
    g_w_o = -d_o.reshape(w_o.shape) + L2 * w_o 
    g_w_o = g_w_o / batchSize
    g_b_o = -np.sum(batchDiff,axis=0) / batchSize
    g_b_o = g_b_o.reshape(b_o.shape)
    g_out = dict()
    g_out['w_o'] = g_w_o
    g_out['b_o'] = g_b_o
    g_out['beta'] = g_beta_o
    g_out['gamma'] = g_gamma_o
    g_model['out'] = g_out
    w_o_T = np.transpose(w_o)
    gemm_norm = linalg.get_blas_funcs(["gemm"],[batchDiff,w_o_T])[0]
    res_diff = gemm_norm(alpha=1.0,a=batchDiff,b=w_o_T,trans_a=False,trans_b=False)
    res_diff = res_diff * mask['mask_res'] / (1-mask['p_fc'])
    start = 0
    for l in range(len(model)-1):
        """
        len(model)-1: number of blocks cascaded.
        reverse order to take the gradients of each layer.
        """
        i = len(model)-2-l
        """
        number of layers (keys) in block i.
        """
        cache_i = cache['block_'+str(i)]
        model_i = model['block_'+str(i)]
        mask_i = mask['block_'+str(i)]
        i_len = len(cache_i)
        """
        block output for the last softmax layer
        """
        block_out_i = cache_i[str(i_len-1)]['layer_out']
        #out_i_len = block_out_i.size // batchSize
        featureNum = block_out_i.shape[3]
        """
        gradient from the softmax layer.
        """
        if i<len(model)-2:
            """
            intermediate layers.
            """
            point = 2
            scale = block_out_i.shape[1] // point
        else:
            """
            last layer.
            """
            point = 2
            scale = block_out_i.shape[1] // point
        #print(res_diff.shape)
        end = start + featureNum * point**2
        out_diff = res_diff[:,start:end]
        start = end
        #out_i_gap = out_diff.size // batchSize
        #scale = int(np.sqrt(out_i_len // (out_i_gap)))
        #scale = block_out_i.shape[1]
        out_shape = block_out_i.shape
        shape = (out_shape[0],out_shape[1]//scale,out_shape[2]//scale,out_shape[3])
        out_diff = out_diff.reshape(shape)
        #print('out_diff for the last conv block')
        #print(out_diff[0,:,:,0])
        out_diff = descent_avgPool(out_diff,scale)
        if i==len(model)-2:
            """
            last block, no maxPooling performed
            """
            #print('diff'+str(out_diff[0:0:4,0:4,0]))
            #print('data input:')
            #print(cache_i['0']['normed'][0,0:4,0:4,0])
            #print('layer_out:')
            #print(cache_i['0']['layer_out'][0,0:4,0:4,0])
            #model_i['1']['sec'][0,0] = model_i['1']['sec'][0,0]- 1e-5
            #print('sec:')
            #print(model_i['0']['sec'])
            #print('slope:')
            #print(model_i['0']['slope'])
            #layer_out_2 = GRelu(cache_i['0']['normed'],model_i['0']['slope'],model_i['0']['sec'])
            #print(np.mean(mask_i['mask']==1))
            #print(layer_out_2[0,0:4,0:4,0])
            #print(np.allclose(layer_out_2,cache_i['0']['layer_out']))
            #print(layer_out_2[0,0:4,0:4,0]-cache_i['0']['layer_out'][0,0:4,0:4,0])
            #print('out_diff:')
            #print(out_diff[0,:,:,0])
            #g_slope,g_sec = gradient_GRelu_para_gpu(gpuarray.to_gpu(out_diff),gpuarray.to_gpu(cache_i['1']['normed']),gpuarray.to_gpu(model_i['1']['slope']),gpuarray.to_gpu(model_i['1']['sec']))
            #g_slope_cpu,g_sec_cpu = gradient_GRelu_para(out_diff,cache_i['1']['normed'],model_i['1']['slope'],model_i['1']['sec'])
            #print('g_sec:')
            #print(g_sec.get())
            #print(g_sec_cpu)
            #print('g_slope')
            #print(g_slope.get())
            #print(g_slope_cpu)
            """
            Take care on the cpu version as it is not correct.
            """
            lower_diff,g_block_i = descent_block(out_diff,mask_i,model_i,cache_i,pad,L2)
            g_model['block_'+str(i)] = g_block_i
            #print('gradient_block:'+str(g_block_i['0']['sec']))
            """
            lower_diff is the gradient after maxPooling from the previous 
            block
            """
        else:
            """
            both maxPooling and Global avergae Pooling.
            """
            poolSign = cache_i[str(i_len-1)]['pool_sign']
            #t0 = timeit.default_timer()
            i_diff = pool_descent(block_out_i,lower_diff,poolSign)
            #t1 = timeit.default_timer()
            #print('max-pool-g:',t1-t0)
            """
            gradient from the softmax layer.
            """
            i_diff += out_diff
            lower_diff,g_block_i = descent_block(i_diff,mask_i,model_i,cache_i,pad,L2)
            g_model['block_'+str(i)] = g_block_i
    """
    For gradient-check purposes.
    """
    g_test = g_model['block_'+'2']['1']['sec']
    #g_test = g_model['out']['w_o']
    momentum = save_to_dict_m(momentum,g_model,1-mu,mu)
    second_order = save_to_dict_second_order(second_order,g_model,1-mv,mv)
    """
    implementation Adam to update model para.
    """
    eps = 1e-8
    model = adam(step,epoch,model,momentum,second_order,mu,mv,alpha,eps,c)
    """
    moving average of model parameters, which is expected to work better than single model, i.e., ensemble can be regarded as a boosting method.
    """
    shadow_model = save_to_dict_m(shadow_model,model,0.001,0.999)
    gc.collect()
    return g_test,model,momentum,second_order,shadow_model
"""
for vector processing convolution
"""
def parallel_conv(batch_map,patchRow):
    featureRow = batch_map.shape[1]
    featureNum = batch_map.shape[3]
    convRow = featureRow - patchRow + 1
    batchSize = batch_map.shape[0]
    conv_map = np.zeros([batchSize,convRow**2,patchRow**2,featureNum])
    for k in range(convRow**2):
        m,n = k//convRow,np.mod(k,convRow)
        conv_map[:,k,:,:] = batch_map[:,m:m+patchRow,n:n+patchRow,:].reshape([batchSize,patchRow**2,featureNum])
    return conv_map
"""
batch processing:convolution
"""
"""
for vector processing convolution
"""
def parallel_conv_2(batch_map,patchRow):
    #batch_map: dim:: batchSize * featureRow * featureRow * featureNum
    featureRow = batch_map.shape[1]
    featureNum = batch_map.shape[3]
    convRow = featureRow - patchRow + 1
    batchSize = batch_map.shape[0]
    conv_map = np.zeros([batchSize,convRow**2,patchRow**2,featureNum])
    for k in range(convRow**2):
        m,n = k//convRow,np.mod(k,convRow)
        conv_map[:,k,:,:] = batch_map[:,m:m+patchRow,n:n+patchRow,:].reshape([batchSize,patchRow**2,featureNum])
    return conv_map
"""
onvolution layer with full connection to all feature maps.
"""
#@profile
def conv_3(data,w_conv,b_conv,pad=True):
    """
    w_conv: patchRow * patchRow * featureNum * featureNum2.
    b_conv: featureNum2 * 1
    full connection for all feature maps (featureNum) from lower layer for each patchRow*patchRow receptive field.
    """
    batchSize = data.shape[0]
    row = data.shape[1]
    col = data.shape[2]
    featureNum = data.shape[3]
    featureNum2 = w_conv.shape[2]
    patchRow = int(np.sqrt(w_conv.shape[0]))
    """
    padding zero on each corner of the feature map from input.
    """
    if pad==True and patchRow==3:
        batchLower = np.zeros((batchSize,row+2,col+2,featureNum))
        batchLower[:,1:-1,1:-1,:] = data
    else:
        batchLower = data
    """
    defaultly a square-shape filter is employed.
    """
    featureRow = int(batchLower.shape[1]) 
    convRow = featureRow-patchRow+1
    batchConv = np.zeros([batchSize,convRow**2,featureNum2])
    batch_map = batchLower.reshape([batchSize,featureRow,featureRow,featureNum])
    conv_map = parallel_conv_2(batch_map,patchRow)
    conv_map = conv_map.reshape([batchSize*convRow**2,patchRow**2*featureNum])
    w_conv_1 = w_conv.reshape([patchRow**2*featureNum,featureNum2])
    convolved = np.zeros([batchSize,convRow**2,featureNum2])
    #conv_map, trans_conv = _force_forder(conv_map)
    #w_conv_1,trans_w = _force_forder(w_conv_1)
    gemm_dot = linalg.get_blas_funcs(["gemm"], (conv_map,w_conv_1))[0]
    temp = gemm_dot(alpha=1.0, a=conv_map, b=w_conv_1,trans_a=False,trans_b=False)
    convolved += temp.reshape(convolved.shape)
    convolved += b_conv
    #batchConv = convolved.reshape([batchSize,convRow**2,featureNum2])
    batchConv = convolved.reshape([batchSize,convRow,convRow,featureNum2])
    #print(batchConv.shape)
    del batchLower
    del temp
    del gemm_dot
    del conv_map
    return batchConv

def conv_3_gpu(data,w_conv,b_conv,pad=True):
    """
    w_conv: patchRow * patchRow * featureNum * featureNum2.
    b_conv: featureNum2 * 1
    full connection for all feature maps (featureNum) from lower layer for each patchRow*patchRow receptive field.
    all of three are numpy arrays.
    """
    batchSize = data.shape[0]
    row = data.shape[1]
    col = data.shape[2]
    featureNum = data.shape[3]
    #print(data.shape)
    """
    padding zero on each corner of the feature map from input.
    """
    patchRow = int(np.sqrt(w_conv.shape[0]))
    if pad==True and patchRow==3:
        batchLower_cpu = np.zeros((batchSize,row+2,col+2,featureNum))
        #print(batchLower_cpu.shape)
        #print(batchLower[:,1:-1,1:-1,:].shape)
        batchLower_cpu[:,1:-1,1:-1,:] = data
        #batchLower = gpuarray.to_gpu(batchLower_cpu)
    else:
        batchLower_cpu = data
        #print(batchLower.shape)
    featureNum2 = w_conv.shape[2]
    """
    defaultly a square-shape filter is employed.
    """
    featureRow = int(batchLower_cpu.shape[1]) 
    convRow = featureRow-patchRow+1
    #batchConv = cumisc.zeros([batchSize,convRow**2,featureNum2],data.dtype)
    batch_map_cpu = batchLower_cpu.reshape([batchSize,featureRow,featureRow,featureNum])
    conv_map_cpu = parallel_conv_2(batch_map_cpu,patchRow)
    conv_map = gpuarray.to_gpu(conv_map_cpu)
    conv_map = conv_map.reshape([batchSize*convRow**2,patchRow**2*featureNum])
    w_conv_1 = w_conv.reshape([patchRow**2*featureNum,featureNum2])
    w_conv_gpu = gpuarray.to_gpu(w_conv_1)
    convolved = culinalg.dot(conv_map,w_conv_gpu)
    b_conv_gpu = gpuarray.to_gpu(b_conv)
    convolved = cumisc.add_matvec(convolved,b_conv_gpu)
    #batchConv = convolved.reshape([batchSize,convRow**2,featureNum2])
    convolved = convolved.reshape([batchSize,convRow,convRow,featureNum2])
    batchConv = convolved.get()
    del batchLower_cpu
    del conv_map
    del w_conv_gpu
    del b_conv_gpu
    del convolved
    gc.collect()
    return batchConv


"""
batch processing: max pooling, with default 2*2 pooling.
"""
def maxPool(batchConv):
    batchSize = batchConv.shape[0]
    featureNum = batchConv.shape[3]
    convRow = batchConv.shape[1]
    poolRow = convRow // 2
    batchConv1 = np.transpose(batchConv,[0,3,1,2])
    batchConv1 = batchConv1.reshape([batchSize*featureNum,convRow,convRow])
    batchPoolSign = np.zeros([batchSize*featureNum,convRow,convRow])
    batchShift = np.zeros([batchSize*featureNum*poolRow**2,2*2])
    batchPool = np.zeros([batchSize*featureNum*poolRow**2,1])
    for j in range(poolRow**2):
        l,k = j//(poolRow),np.mod(j,poolRow)
        step = poolRow**2
        batchShift[j::step,:] = batchConv1[:,l*2:(l+1)*2,k*2:(k+1)*2].reshape([batchSize*featureNum,2*2])
    batchPool = np.max(batchShift, axis=1).reshape([batchSize*featureNum*poolRow**2,1])
    indexList = np.argmax(batchShift,axis=1)
    batchShiftSign = np.zeros(batchShift.shape)
    batchShiftSign[range(batchSize*featureNum*poolRow**2),indexList] = 1
    for j in range(poolRow**2):
        l,k = j//(poolRow),np.mod(j,poolRow)
        step = poolRow**2
        batchPoolSign[:,l*2:(l+1)*2,k*2:(k+1)*2] = batchShiftSign[j::step,:].reshape([batchSize*featureNum,2,2])
    batchPoolSign = batchPoolSign.reshape(batchSize,featureNum,convRow,convRow)
    batchPoolSign = np.transpose(batchPoolSign,[0,2,3,1])
    batchPool = batchPool.reshape([batchSize,featureNum,poolRow,poolRow])
    batchPool = np.transpose(batchPool,[0,2,3,1])
    del batchShift
    del batchShiftSign
    del batchConv1
    return batchPool,batchPoolSign
"""
average pooling layer
If used for global average pooling,
just set the scale to be the row/col of the feature map.
"""
def avgPool(batchConv,scale):
    batchSize = batchConv.shape[0]
    featureNum = batchConv.shape[3]
    convRow = batchConv.shape[1]
    poolRow = convRow // scale
    batchConv1 = np.transpose(batchConv,[0,3,1,2])
    batchConv1 = batchConv1.reshape([batchSize*featureNum,convRow,convRow])
    batchShift = np.zeros([batchSize*featureNum*poolRow**2,scale**2])
    batchPool = np.zeros([batchSize*featureNum*poolRow**2,1])
    for j in range(poolRow**2):
        l,k = j//(poolRow),np.mod(j,poolRow)
        step = poolRow**2
        batchShift[j::step,:] = batchConv1[:,l*scale:(l+1)*scale,k*scale:(k+1)*scale].reshape([batchSize*featureNum,scale**2])
    batchPool = np.mean(batchShift, axis=1).reshape([batchSize*featureNum*poolRow**2,1])
    batchPool = batchPool.reshape([batchSize,featureNum,poolRow,poolRow])
    batchPool = np.transpose(batchPool,[0,2,3,1])
    del batchShift
    del batchConv1
    return batchPool
"""
descent to the avgPool layer.
"""
def descent_avgPool(diff,scale):
    shape = diff.shape
    lower_diff = np.zeros([shape[0],shape[1]*scale,shape[2]*scale,shape[3]])
    for m in range(scale):
        for n in range(scale):
            lower_diff[:,m::scale,n::scale,:] = diff / scale**2
    return lower_diff

"""
batch normalization layer:
The normalization layer is to get the data out of the saturation state and
make it potential for further performance improvement.
"""    
def norm_layer(batchData,beta,gamma):
    tiny = 1e-20
    batchSize = batchData.shape[0]
    shape = batchData.shape
    if len(shape)>2:
       batchSize2 = batchSize * shape[1]**2
    else:
       batchSize2 = batchSize
    size = batchData.size
    paraNum = size // batchSize2
    batchShifted = batchData.reshape([batchSize2,paraNum])
    batchMean = np.mean(batchShifted,axis=0).reshape([1,paraNum])
    batchStd = np.std(batchShifted,axis=0).reshape([1,paraNum])
    batchNorm = np.divide(np.subtract(batchShifted,batchMean),batchStd+tiny)
    batchNorm = beta * batchNorm + gamma
    batchNorm = batchNorm.reshape(shape)
    del batchShifted
    return batchNorm, batchMean, batchStd

def create_dict(layer_in,convolved,normed,norm_mean,norm_std,layer_out):
    cache = dict()
    cache['layer_in'] = layer_in
    cache['convolved'] = convolved
    cache['normed'] = normed
    cache['norm_mean'] = norm_mean
    cache['norm_std'] = norm_std
    cache['layer_out'] = layer_out
    return cache

def conv_bn_activation(layer_in,mask,para,pad=True,train=True):
    """
    data: a 4d tensor input to the conv layer.
    para: a dict obj containing all necessary parameters, such as
    w_conv,b_conv,BN parameters, as well as activation function parameters.
    """
    #convolved_cpu = conv_3(layer_in,para['w_conv'],para['b_conv'],pad)
    convolved = conv_3_gpu(layer_in,para['w_conv'],para['b_conv'],pad)
    #print('conv:', np.allclose(convolved_cpu,convolved))
    normed, norm_mean, norm_std = norm_layer(convolved,para['beta'],para['gamma'])
    activated = Relu(normed)
    if train==True:
        layer_out = activated * mask['mask'] / (1 - mask['p_conv'])
    else:
        layer_out = activated
    cache = create_dict(layer_in,convolved,normed,norm_mean,norm_std,layer_out)
    gc.collect()
    return cache

def block(block_in,mask,model,pad=True,train=True):
    """
    mask: dropout mask.
    model: model parameters for this block.
    input: a 4d tensor feed into this block.
    cache: returned values of the output of each layer within this block.
    """
    depth = len(model)
    #print('layer number in block: %f' %(depth))
    layer_in = block_in
    cache = dict()
    for i in range(depth):
        para = model[str(i)]
        cache_i = conv_bn_activation(layer_in,mask,para,pad,train)
        layer_in = cache_i['layer_out']
        cache[str(i)] = cache_i
    """
    output the activation of this block as well as the cached intermediate data.
    It is noted that block-activation is also in the cache, but for convenient purpose,
    we output it together with block-level cache.
    """
    block_out = layer_in
    return block_out,cache

def create_g_dict(g_w_conv,g_b_conv,g_beta,g_gamma):
    g_para = dict()
    g_para['w_conv'] = g_w_conv
    g_para['b_conv'] = g_b_conv
    g_para['beta'] = g_beta
    g_para['gamma'] = g_gamma
    return g_para

def descent_conv_bn_activation(diff,cache,mask,para,pad,L2):
    diff = diff * mask['mask'] / ( 1 - mask['p_conv'])
    df = gradient_Relu(cache['normed'])
    diff = diff * df
    conv_diff,g_beta,g_gamma = norm_descent(diff,cache['convolved'],cache['normed'],cache['norm_mean'],cache['norm_std'],para['beta'],para['gamma'],L2)
    g_w_conv,g_b_conv,lower_diff = conv_descent_3_gpu(cache['layer_in'],cache['convolved'],para['w_conv'],para['b_conv'],conv_diff,L2,pad)
    #g_w_conv_cpu,g_b_conv_cpu,lower_diff_cpu = conv_descent_3(cache['layer_in'],cache['convolved'],para['w_conv'],para['b_conv'],conv_diff,L2,pad)
    #print('descent-g-conv:', np.allclose(g_w_conv_cpu,g_w_conv))
    #print('descent-b-conv:', np.allclose(g_b_conv_cpu,g_b_conv))
    #print('descent-diff-conv:', np.allclose(lower_diff,lower_diff_cpu) )
    g_para = create_g_dict(g_w_conv,g_b_conv,g_beta,g_gamma)
    gc.collect()
    return lower_diff,g_para

def descent_block(diff,mask,model,cache,pad,L2):
    """
    diff: from upper layers
    mask: dropout mask.
    model: model parameters for this block.
    cache: intermediate values in-between this block.
    """
    #n = 5
    depth = len(model)
    g_model = dict()
    diff_i = diff
    for l in range(depth):
        """
        reverse order for backpropagation.
        """
        i = depth-(l+1)
        para_i = model[str(i)]
        cache_i = cache[str(i)]
        lower_diff,g_para_i = descent_conv_bn_activation(diff_i,cache_i,mask,para_i,pad,L2)
        diff_i = lower_diff
        g_model[str(i)] = g_para_i
    return lower_diff,g_model
    
"""
batch processing main course: forward-propogation
"""
##@profile
def batch(batchData,batchLabel,model,L2,mask,train=True):
    tiny = 1e-10
    batchSize = batchData.shape[0]
    if train==True:
        p_v = mask['p_v']
        mask_v = mask['mask_v'] 
        batchData = batchData * mask_v / (1-p_v)
    """
    raw pixel with dropout.
    """
    cache = dict()
    batchRes = avgPool(batchData,8)
    batchRes = batchRes.reshape((batchSize,batchRes.size//batchSize))
    """
    normally we only consider the output of the last layer of each block, but it would be
    interesting to include all layers in the final softmax layer for detection.
    """
    batchSize = batchLabel.shape[0]
    """
    model is a dict with one key for a building block as well as
    a key referring to the last full-connected layer feeding into the softmax layer.
    """
    block_in = batchData
    for i in range(len(model)-1):
        block_para = model['block_'+str(i)]
        #print(block_para.keys())
        if train==True:
            block_mask = mask['block_'+str(i)]
        else:
            block_mask=[]
        pad = True
        #t0 = timeit.default_timer()
        block_out_i,cache_i = block(block_in,block_mask,block_para,pad,train)
        #t1 = timeit.default_timer()
        #print('block:', t1-t0)
        last = len(cache_i)-1
        #print(last)
        #print(len(cache_i))
        if i<len(model)-2:
            """
            in-between each block, 2*2 
            maxPool is used to reduce the size
            of the feature map.
            insert pool parameters into the cache
            of each block except the last block, which is directly feed
            into the last layer after global average pooling. 
            """
            #t0=timeit.default_timer()
            block_in,block_sign = maxPool(block_out_i)
            #t1=timeit.default_timer()
            #print('max-pool:', t1-t0)
            cache_i[str(last)]['pool'] = block_in
            cache_i[str(last)]['pool_sign'] = block_sign
        cache['block_'+str(i)] = cache_i
        if i<len(model)-2:
            """
            keep 1 points per feature map for hidden layers.
            """
            scale_i = block_out_i.shape[1] // 2
        else:
            """
            keep 1 points per feature map for the last layer.
            reserve enough information for the final convolved layers.
            """
            scale_i = block_out_i.shape[1] // 2  
        """
        global average pooling is applied to the output of one block. 
        but can be on all layers.
        In factt, if only the block-level output is used in the final layer
        for detection, a fine-granity scale for 
        avg pooling will be used for performance comparison.
        In addition, it is kindly noted that
        here we use the output of each block(before maxPooling)
        for average pooling to reserve some minor information, which
        is expected to work better, from my own personal point of view.
        """
        #scale_i = scale_i // 2 #reserve more information for final detection.
        avg_block_out_i = avgPool(block_out_i,scale_i)
        cache['block_'+str(i)][str(last)]['avgPool'] = avg_block_out_i
        cache['block_'+str(i)][str(last)]['avgPool_scale'] = scale_i
        size_i = avg_block_out_i.size
        avg_block_out_i = avg_block_out_i.reshape(batchSize,size_i//batchSize)
        batchRes = np.concatenate((avg_block_out_i,batchRes),axis=1)
    if train == True:
        p_fc = mask['p_fc']
        norm_res = mask['mask_res']
        batchRes = batchRes * norm_res / (1-p_fc)
    """
    take all previous activation into final softmax layer. 
    """
    w_o = model['out']['w_o']
    b_o = model['out']['b_o']
    beta_o = model['out']['beta']
    gamma_o = model['out']['gamma']
    gemm = linalg.get_blas_funcs(["gemm"],[batchRes,w_o])[0]
    #print('w_o shape:'+str(w_o.shape))
    #print('batchRes shape:'+str(batchRes.shape))
    temp = gemm(alpha=1.0,a=batchRes,b=w_o,trans_a=False,trans_b=False)
    batchC = temp.reshape([batchSize,w_o.shape[1]]) + b_o
    outNormed,outMean,outStd = norm_layer(batchC,beta_o,gamma_o)
    #batchC = np.subtract(batchC, np.mean(batchC, axis=1).reshape([batchSize,1]))
    batchProb = np.exp(outNormed)
    """
    1e-20 is a value to guarantee there is no
    leakage event happens, if all values tend to be zero.
    """
    #eps = 1e-30
    #numClass = batchProb.shape[1]
    batchProb = np.divide(batchProb, np.sum(batchProb,axis=1).reshape([batchSize,1]))
    cache_softmax = dict() 
    cache_softmax['norm'] = batchC
    cache_softmax['normed'] = outNormed
    cache_softmax['norm_mean'] = outMean
    cache_softmax['norm_std'] = outStd
    cache_softmax['layer_in'] = batchRes
    cache_softmax['layer_out'] = batchProb
    batchPredictLabel = np.argmax(batchProb,axis=1).reshape([batchSize,1])
    batchComp = np.equal(batchPredictLabel,batchLabel)
    loss = - np.mean(np.log(batchProb+tiny)[np.arange(batchSize),batchLabel.reshape([1,batchSize])])    
    if np.isnan(loss):
        print('error! loss is not a number!')
        print(str(np.where(batchProb==0)))
    regulator = get_regulator(L2,model) / batchSize
    #regulator = L2 * ( np.sum(w_conv**2) + np.sum(w_conv_2**2) + np.sum(w_conv_3**2) + np.sum(w_o**2) ) / batchSize
    cache_softmax['loss'] = loss
    cache_softmax['regulator'] = 0.5 * regulator
    cache_softmax['result'] = batchComp
    cache['out'] = cache_softmax
    total_loss = loss + 0.5 * regulator
    if regulator>100:
        """
        sth wrong happend, as the L2 norm of the whole parameters is too large
        """
        #print(w_conv[0,0,:])
        print('total loss is %f and actual loss is %f.\n' %(total_loss,loss))
    gc.collect()
    return cache,total_loss,batchComp
"""
regulation
"""
def get_regulator(L2,model):
    result = 0
    for block in model.keys():
        block_para = model[block]
        if block =='out':
            result += np.sum(block_para['w_o']**2)
        else:
            for layer in block_para.keys():
                layer_para = block_para[str(layer)]
                result += np.sum(layer_para['w_conv']**2)
                """
                normalization para are not put into regulation.
                """
    result *= L2
    return result
"""
error rate per type
"""
def correctRate(dataComp, dataLabels):
    correctCount = np.zeros([10,2], dtype = float)
    for index, label in zip(dataComp, dataLabels):
        #correctCount[label,0] += correctCount[label,0];
        if index == 0:
            correctCount[label,0] += correctCount[label,0]
        else:
            correctCount[label,1] += correctCount[label,1]
    return correctRate
"""
create a dict sotring lists of values before activation.
"""
def sec_init(model):
    sec = dict()
    for key in model.keys():
        if key!='out':
           sec[key] = dict()
           for key2 in model[key].keys():
               sec[key][key2] = np.empty([0,1],dtype='float64')
    return sec
def sec_add(sec,cache):
    for key in cache.keys():
                if key!='out':
                   for key2 in cache[key].keys():
                       c1 = cache[key][key2]['normed']
                       i_size = c1.size
                       sec[key][key2] = np.append(sec[key][key2],c1.reshape([i_size,1]))
    return sec
def model_sec(model,sec):
    for key in sec.keys():
        for key2 in sec[key].keys():
            sec[key][key2] = np.sort(sec[key][key2])
            i_size = sec[key][key2].size
            model[key][key2]['sec'][0,0] = sec[key][key2][int(i_size * 0.5)]
            model[key][key2]['sec'][0,1] = sec[key][key2][int(i_size * 0.4)]   
            model[key][key2]['sec'][1,0] = sec[key][key2][int(i_size * 0.6)]
            model[key][key2]['sec'][1,1] = sec[key][key2][int(i_size * 0.7)]
            print(key + key2)
            print(model[key][key2]['sec'])
    return model
"""
Interface to the outside world.
"""
def interface(batchSize, epochs):
    print('... Loading data')    
    start = timeit.default_timer()
    trainData, trainLabel, testData, testLabel, dict_names = LoadData()
    finish = timeit.default_timer()
    print('preprocessing uses %f seconds.' %(finish-start))
    print(trainData.shape)
    """
    plt.imshow(trainData[99].astype('uint8'))
    print(trainLabel.shape)
    print(trainLabel[99])
    plt.title(dict_names[int(trainLabel[99])])
    plt.axis('off')
    plt.show()
    """    
    model,L2,momentum,second_order,mask_shape,shadow_model = Initial(batchSize)
    #model,L2,momentum,second_order,mask_shape,shadow_model = Load_model(batchSize)
    alpha = 1e-2
    print('... training the model')
    start_time = timeit.default_timer()
    epoch = 0
    done_Loop = False
    n_train = trainLabel.shape[0] // batchSize
    n_test = testLabel.shape[0] // batchSize
    train_rate = np.zeros([epochs+1,1])
    #val_rate = np.zeros([epochs+1,1])
    test_rate = np.zeros([epochs+1,1])
    #rate = 0
    step = epoch * 250
    #prior_rate = 0
    best_rate = 0
    model_info = dict()
    rate = dict()
    """
    indicating whether learning rate is appropriate.
    """
    #sign = 0
    sec = sec_init(model)
    for index in range(5):
        batchData = trainData[index*batchSize:(index+1)*batchSize]
        batchLabel = trainLabel[index*batchSize:(index+1)*batchSize]
        cache,loss,batchComp = batch(batchData,batchLabel,model,L2,mask=[],train=False)
        sec = sec_add(sec,cache)
        model = model_sec(model,sec)
    del sec 
    while epoch < epochs and not done_Loop:
        epoch = epoch + 1
        print('epoch '+str(epoch)+' begins:')
        epoch_start = timeit.default_timer()
        #best_rate = 0
        t1 = timeit.default_timer()
        val_epoch = 0
        if epoch<val_epoch:
            n_val = 50
            """
            validation for tuning hyper-parameters.
            """
        else:
            n_val = 0 
            """
            used for training purposes.
            """
        trainData[0:-n_val*batchSize,], trainLabel[0:-n_val*batchSize,] = permutation(trainData[0:-n_val*batchSize,],trainLabel[0:-n_val*batchSize,])
        for index in range(n_train-n_val):
            step += 1
            batchData = trainData[index*batchSize:(index+1)*batchSize]
            batchLabel = trainLabel[index*batchSize:(index+1)*batchSize]
            model,batchComp,loss,momentum,second_order,shadow_model = trainBatch(step,alpha,batchData,batchLabel,model,shadow_model,L2,momentum,second_order,epoch,mask_shape)
            #print(model['block_2']['0']['sec'])
            #print(model['block_2']['1']['sec'])
            if index>=0 and np.mod(index,10)==0:
                print('current index is: ' + str(index))
                current_rate = np.mean(batchComp)
                print('correct rate is %f,' % current_rate)
                print('current loss is %f,' %loss)
                #regulator = 0.5 * L2 * (np.sum(w_conv**2)+np.sum(w_conv_2**2)++np.sum(w_conv_3**2)+np.sum(w_h**2)+np.sum(w_o**2)) / batchSize
                #print('current loss without regularization is %f,' %loss-regulator)
            if index>=0 and np.mod(index,10)==0:
                t2 = timeit.default_timer()
                print(' 10 training batches %d uses %f seconds,' % (index+1, t2-t1))
        epoch_end = timeit.default_timer()
        print('Epoch %d uses %f seconds,' %(epoch, epoch_end-epoch_start))
        model_start = timeit.default_timer()         
        train_prediction = np.empty([0,1],dtype=float)
        train_loss = 0
        batchSize2 = batchSize
        n_train2 = trainLabel.shape[0] // batchSize2
        if epoch <= val_epoch:
            n_val2 = 5000//batchSize2
        else:
            n_val2 = 0
        #for index in range(1):
        n_train2 = 0
        """
        saving time in not computing train set accuracy.
        """
        for index in range(n_train2-n_val2):
            batchData = trainData[index*batchSize2:(index+1)*batchSize2]
            batchLabel = trainLabel[index*batchSize2:(index+1)*batchSize2]
            cache,loss,batchComp = batch(batchData,batchLabel,model,L2,mask=[],train=False)
            train_loss += loss
            if np.isnan(loss) or not np.isfinite(loss):
                print('L2 is %f,' % (L2))
                #print(batchProb[0,])
                return
            train_prediction = np.append(train_prediction,batchComp.reshape([batchSize2,1]),axis=0)
        #print('the loss for train data is %f \n ,' % (train_loss / (n_train2-n_val2)))
        if epoch <= val_epoch:
            val_prediction = np.empty([0,1],dtype=float)
            val_loss = 0
            #for index in range(1):
            for index in range(n_train2-n_val2,n_train2):
                batchData = trainData[index*batchSize2:(index+1)*batchSize2]
                batchLabel = trainLabel[index*batchSize2:(index+1)*batchSize2]
                cache,loss,batchComp = batch(batchData,batchLabel,model,L2,mask=[],train=False)
                val_prediction = np.append(val_prediction,batchComp.reshape([batchSize2,1]),axis=0)
                val_loss += loss
            print(val_prediction.shape)        
            print('the loss for validation data is %f \n ,' % (val_loss/n_val2))
            print('total correctly classfied objects and its mean value in validation dataset: %f and %f \n,' %(np.sum(val_prediction),np.mean(val_prediction)))
            if np.isnan(loss) or not np.isfinite(loss):
                #print(batchProb[0,])
                print(loss)
                return
            """
            adjust regularization level if it is larger than 0.01.
            """
            if np.abs(np.mean(val_prediction)-np.mean(train_prediction))>0.01:
                if np.mean(val_prediction)>np.mean(train_prediction):
                    """
                    large L2, can be decayed to combat underfitting.
                    """
                    L2 = L2 * 0.99
                else:
                    """
                    improve regulation level to combat overfitting.
                    """
                    L2 = L2 * 1.01
        """
        too much computation overhead right now. but will
        be checked later with sufficient computation resources
        i.e., for the small MNIST dataset.
        """
        test_prediction = np.empty([0,1],dtype=float)
        test_loss = 0
        n_test = testLabel.shape[0] // batchSize2
        print(model['block_2']['0']['sec'])                                   
        print(model['block_2']['1']['sec'])
        for index in range(n_test):
            batchData = testData[index*batchSize2:(index+1)*batchSize2]
            batchLabel = testLabel[index*batchSize2:(index+1)*batchSize2]
            cache,loss,batchComp = batch(batchData,batchLabel,model,L2,mask=[],train=False)
            test_prediction = np.append(test_prediction,batchComp.reshape([batchSize2,1]),axis=0)
            test_loss += loss
        print(test_prediction.shape)        
        print('the loss for test data is %f \n ,' % (test_loss/n_test))
        print('total correctly classfied objects in test dataset: %f \n,' %(np.sum(test_prediction)))
        if np.isnan(loss) or not np.isfinite(loss):
            print(test_loss)
            return 
        model_end = timeit.default_timer()
        print('model prediction time for all data is %f,' % (model_end-model_start))
        print('test data prediction rate is %f,' %(np.mean(test_prediction)))
        current_rate = np.mean(test_prediction)
        #train_rate[(epoch-1)] = np.mean(train_prediction)
        test_rate[(epoch-1)] = np.mean(test_prediction)
        rate['test'] = test_rate
        if best_rate < current_rate:
            best_rate = current_rate
            best_model = model
            model_info['rate'] = rate
            model_info['model'] = best_model
            model_info['shadow_model'] = shadow_model
            model_info['mask_shape'] = mask_shape
            model_info['second_order'] = second_order
            model_info['momentum'] = momentum
            model_info['best_rate'] = best_rate
            save_file(model_info)
        """
        learning rate update
        """
        print('best rate until recently is: ', best_rate)
        alpha = learning_rate(epoch)
        #prior_rate = current_rate
        print('current learning rate:%f \n' %(alpha))
        
    end_time = timeit.default_timer()
    print('time for model building: %f,' % (end_time - start_time) )
    print('best rate is %f ,' %(best_rate) )
    #rate = dict()
    #rate['train'] = train_rate
    rate['test'] = test_rate
    test_prediction = np.empty([0,1],dtype=float)
    test_loss = 0
    """
    test rate.
    """
    batchSize2 = batchSize
    n_test = testLabel.shape[0] // batchSize2
    #for index in range(1):
    for index in range(n_test):
        batchData = testData[index*batchSize2:(index+1)*batchSize2]
        batchLabel = testLabel[index*batchSize2:(index+1)*batchSize2]
        cache,loss,batchComp = batch(batchData,batchLabel,shadow_model,L2,mask=[],train=False)
        test_prediction = np.append(test_prediction,batchComp.reshape([batchSize2,1]),axis=0)
        test_loss += loss
    print(test_prediction.shape)        
    print('the loss for test data is %f \n ,' % (test_loss/n_test))
    print('total correctly classfied objects in test dataset: %f \n,' %(np.sum(test_prediction)))
    print('the number of correctly classfied in the first 50 objects in test dataset is %f \n' %(np.sum(test_prediction[:50])))        
    if np.isnan(loss) or not np.isfinite(loss):
        print(test_loss)
        return
    #model_info = dict()
    model_info['rate'] = rate
    model_info['model'] = best_model
    model_info['shadow_model'] = shadow_model
    model_info['mask_shape'] = mask_shape
    model_info['second_order'] = second_order
    model_info['momentum'] = momentum
    model_info['best_rate'] = best_rate
    save_file(model_info)
    gc.collect()
    return rate 

def learning_rate(epoch):
    lr = 1e-2
    if epoch>=1 and epoch<=10:
        lr = 2 * 1e-3
    elif epoch>10 and epoch<=35:
        lr = 1e-3
    elif epoch>35 and epoch<=60:
        lr = 1e-4
    elif epoch>60 and epoch<=75:
        lr = 1e-5
    elif epoch>75:
        lr = 1e-6
    print('current learning rate is: %f \n' %(lr))
    return lr
    
if __name__ == '__main__':
    gc.enable()
    rate = interface(batchSize=200, epochs=85)
