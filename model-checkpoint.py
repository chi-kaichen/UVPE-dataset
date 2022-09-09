import os
import time
import cv2
import vgg19
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, Reshape, GlobalAveragePooling2D, Activation

from scipy import misc
#from skimage.measure import compare_psnr
#from skimage.measure import compare_ssim

from operations import TransposeConv, DropOut, AvgPool
from operations import Conv, Conv7, Conv5, Conv3, Conv1, ReLU, LeakyReLU, BatchNorm, max_pool_2x2, Global_Average_Pooling, Fully_connected

from tensorflow.keras import layers,Sequential,regularizers


class GAN():

    def __init__(self, args):   #__init__方法的第一个参数永远是self，表示创建的实例本身，因此，在__init__方法内部，就可以把各种属性绑定到self，因为self就指向创建的实例本身。
        self.num_discriminator_filters = args.D_filters
        self.layers = args.layers
        self.growth_rate = args.growth_rate
        self.gan_wt = args.gan_wt
        self.l1_wt = args.l1_wt
        self.vgg_wt = args.vgg_wt
        self.restore = args.restore
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.model_name = args.model_name
        self.decay = args.decay
        self.save_samples = args.save_samples
        self.sample_image_dir = args.sample_image_dir
        self.A_dir = args.A_dir
        self.B_dir = args.B_dir
        self.wb_dir = args.wb_dir
        #self.gc_dir = args.gc_dir
        self.custom_data = args.custom_data
        self.val_fraction = args.val_fraction
        self.val_threshold = args.val_threshold
        self.logger_frequency = args.logger_frequency
        
        self.EPS = 10e-8
        self.score_best = -1
        self.ckpt_dir = os.path.join(os.getcwd(), self.model_name, 'checkpoint')
        self.tensorboard_dir = os.path.join(os.getcwd(), self.model_name, 'tensorboard')


    def REM1(self, input_, name):
        with tf.variable_scope(name):                     
            REM1_conv1 = Conv3(input_, 16, stride = 1, name = 'REM1_Conv1')
            REM1_conv2 = LeakyReLU(REM1_conv1)
            REM1_conv3 = Conv3(REM1_conv2, 16, stride = 1, name = 'REM1_Conv2')
            REM1_conv4 = LeakyReLU(REM1_conv3)
            REM1_conv5 = Conv3(REM1_conv4, 16, stride = 1, name = 'REM1_Conv3')                  
#            REM1_conv6 = tf.add(input_, REM1_conv5)
            REM1_conv6 = input_ + REM1_conv5               
            REM1_conv7 = Conv3(REM1_conv6, 16, stride = 1, name = 'REM1_Conv4')
            REM1_conv8 = LeakyReLU(REM1_conv7)
            REM1_conv9 = Conv3(REM1_conv8, 16, stride = 1, name = 'REM1_Conv5')
            REM1_conv10 = LeakyReLU(REM1_conv9)
            REM1_conv11 = Conv3(REM1_conv10, 16, stride = 1, name = 'REM1_Conv6')                 
#                     REM1_conv12 = tf.add(REM1_conv6, REM1_conv11)
            REM1_conv12 = REM1_conv6 + REM1_conv11
                     
        return REM1_conv12


    def REM2_and_REM3(self, input_, name):
        with tf.variable_scope(name):
            REM2_conv1 = Conv3(input_, 16, stride = 1, name = 'REM2_Conv1')
            REM2_conv2 = LeakyReLU(REM2_conv1)
            REM2_conv3 = Conv3(REM2_conv2, 16, stride = 1, name = 'REM2_Conv2')
            REM2_conv4 = LeakyReLU(REM2_conv3)
            REM2_conv5 = Conv3(REM2_conv4, 16, stride = 1, name = 'REM2_Conv3')
#            REM2_conv6 = tf.add(input_, REM2_conv5)
            REM2_conv6 = input_ + REM2_conv5                 
            REM2_conv7 = Conv3(REM2_conv6, 16, stride = 1, name = 'REM2_Conv4')
            REM2_conv8 = LeakyReLU(REM2_conv7)
            REM2_conv9 = Conv3(REM2_conv8, 16, stride = 1, name = 'REM2_Conv5')
            REM2_conv10 = LeakyReLU(REM2_conv9)
            REM2_conv11 = Conv3(REM2_conv10, 16, stride = 1, name = 'REM2_Conv6')
#                     REM23_conv12 = tf.add(REM23_conv6, REM23_conv11)
            REM2_conv12 = REM2_conv6 + REM2_conv11
            REM3_conv1 = Conv3(REM2_conv12, 16, stride = 1, name = 'REM3_Conv1')
            REM3_conv2 = LeakyReLU(REM3_conv1)
            REM3_conv3 = Conv3(REM3_conv2, 16, stride = 1, name = 'REM3_Conv2')
            REM3_conv4 = LeakyReLU(REM3_conv3)
            REM3_conv5 = Conv3(REM3_conv4, 16, stride = 1, name = 'REM3_Conv3')
            REM3_conv6 = REM2_conv12 + REM3_conv5
            REM3_conv7 = Conv3(REM3_conv6, 16, stride = 1, name = 'REM3_Conv4')
            REM3_conv8 = LeakyReLU(REM3_conv7)
            REM3_conv9 = Conv3(REM3_conv8, 16, stride = 1, name = 'REM3_Conv5')
            REM3_conv10 = LeakyReLU(REM3_conv9)
            REM3_conv11 = Conv3(REM3_conv10, 16, stride = 1, name = 'REM3_Conv6')
            REM3_conv12 = REM3_conv6 + REM3_conv11
                     
        return REM3_conv12
    
    
    def SE(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = Global_Average_Pooling(input_x)
            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = tf.nn.relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = tf.sigmoid(excitation)
            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation
            return scale


# with 语句，不管在处理文件过程中是否发生异常，都能保证 with 语句执行完毕后已经关闭了打开的文件句柄。
    def generator(self):
        
        with tf.variable_scope('Inputconv',reuse=tf.AUTO_REUSE):          

            T_in_2down = max_pool_2x2(self.images_wb)  #介质传输 128*128
            T_in_4down = max_pool_2x2(T_in_2down)      #介质传输 64*64
            
            C2 = max_pool_2x2(self.RealA)  #水下图像 128*128
            C3 = max_pool_2x2(C2)  #水下图像 64*64
            C4 = max_pool_2x2(C3)  #水下图像 32*32
            
            #32*32 patch
            conv32_1 = Conv3(C4, 16, stride = 1, name = '32patch_conv1')
            conv32_2 = LeakyReLU(conv32_1)
            conv32_3 = self.REM1(conv32_2, name = '32patch_conv2') #32*32*16
            
            #ConvLSTM参考Single Image Deraining Network with Rain Embedding Consistency and Layered LSTM (ECNet)
            i_32 = tf.layers.conv2d(conv32_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                    kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_32 = tf.layers.conv2d(conv32_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                    kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_32 = tf.layers.conv2d(conv32_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                    kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_32 = tf.layers.conv2d(conv32_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                    kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            #print("i_32", np.shape(i_32))
            
            i_32 = tf.sigmoid(i_32)
            f_32 = tf.sigmoid(f_32)
            g_32 = tf.nn.tanh(g_32)
            o_32 = tf.sigmoid(o_32)
            
            c_32 = i_32 * g_32
            h_32 = o_32 * tf.nn.tanh(c_32)
            
            conv32_4 = tf.concat(axis = 3, values = [conv32_3, h_32])
            
            conv32_5 = Conv3(conv32_4, 16, stride = 1, name = '32patch_conv3')
            conv32_output1 = LeakyReLU(conv32_5)
#            conv32_output1 = self.REM1(conv32_6, name = '32patch_conv4')
            
#            conv32_7 = tf.add(C4, conv32_output1)  #有的论文是add 有的是concat
            conv32_7 = conv32_2 + conv32_output1
            conv32_8 = Conv3(conv32_7, 16, stride = 1, name = '32patch_conv5')
            conv32_9 = LeakyReLU(conv32_8)
            conv32_10 = self.REM1(conv32_9, name = '32patch_conv6') #32*32*16
            
            i_32_2 = tf.layers.conv2d(conv32_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_32_2 = tf.layers.conv2d(conv32_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_32_2 = tf.layers.conv2d(conv32_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_32_2 = tf.layers.conv2d(conv32_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_32_2 = tf.sigmoid(i_32_2)
            f_32_2 = tf.sigmoid(f_32_2)
            g_32_2 = tf.nn.tanh(g_32_2)
            o_32_2 = tf.sigmoid(o_32_2)
            
            c_32_2 = f_32_2 * c_32 + i_32_2 * g_32_2
            h_32_2 = o_32_2 * tf.nn.tanh(c_32_2)
            conv32_11 = tf.concat(axis = 3, values = [conv32_10, h_32_2])
            
            conv32_12 = Conv3(conv32_11, 16, stride = 1, name = '32patch_conv7')
            conv32_output2 = LeakyReLU(conv32_12)
#            conv32_output2 = self.REM2_and_REM3(conv32_13, name = '32patch_conv8')
            
#            conv32_14 = tf.add(C4, conv32_output2)
            conv32_14 = conv32_2 + conv32_output2
            conv32_15 = Conv3(conv32_14, 16, stride = 1, name = '32patch_conv9')
            conv32_16 = LeakyReLU(conv32_15)
            conv32_17 = self.REM1(conv32_16, name = '32patch_conv10')
            
            i_32_3 = tf.layers.conv2d(conv32_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_32_3 = tf.layers.conv2d(conv32_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_32_3 = tf.layers.conv2d(conv32_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_32_3 = tf.layers.conv2d(conv32_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_32_3 = tf.sigmoid(i_32_3)
            f_32_3 = tf.sigmoid(f_32_3)
            g_32_3 = tf.nn.tanh(g_32_3)
            o_32_3 = tf.sigmoid(o_32_3)
            
            c_32_3 = f_32_3 * c_32_2 + i_32_3 * g_32_3
            h_32_3 = o_32_3 * tf.nn.tanh(c_32_3)
            conv32_18 = tf.concat(axis = 3, values = [conv32_17, h_32_3])
            
            conv32_19 = Conv3(conv32_18, 16, stride = 1, name = '32patch_conv11')
            conv32_20 = LeakyReLU(conv32_19)
            conv32_output3 = self.REM1(conv32_20, name = '32patch_conv12')
            
            
            #64*64 patch
            conv64_1 = Conv3(C3, 16, stride = 1, name = '64patch_conv1')
            conv64_2 = LeakyReLU(conv64_1)
            conv64_3 = self.REM1(conv64_2, name = '64patch_conv2')
            
            i_64_1 = tf.layers.conv2d(conv64_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_64_1 = tf.layers.conv2d(conv64_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_64_1 = tf.layers.conv2d(conv64_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_64_1 = tf.layers.conv2d(conv64_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_64_1 = tf.sigmoid(i_64_1)
            f_64_1 = tf.sigmoid(f_64_1)
            g_64_1 = tf.nn.tanh(g_64_1)
            o_64_1 = tf.sigmoid(o_64_1)
            c_64_1 = i_64_1 * g_64_1
            h_64_1 = o_64_1 * tf.nn.tanh(c_64_1)
            conv64_4 = tf.concat(axis = 3, values = [conv64_3, h_64_1])
            
            conv64_5 = Conv3(conv64_4, 16, stride = 1, name = '64patch_conv3')
            conv64_output1 = LeakyReLU(conv64_5)
#            conv64_output1 = self.REM1(conv64_6, name = '64patch_conv4')
            
#            conv64_7 = tf.add(C3_2, conv64_output1)
            conv64_7 = conv64_2 + conv64_output1
            conv64_8 = Conv3(conv64_7, 16, stride = 1, name = '64patch_conv5')
            conv64_9 = LeakyReLU(conv64_8)
            conv64_10 = self.REM1(conv64_9, name = '64patch_conv6')
            
            i_64_2 = tf.layers.conv2d(conv64_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_64_2 = tf.layers.conv2d(conv64_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_64_2 = tf.layers.conv2d(conv64_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_64_2 = tf.layers.conv2d(conv64_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_64_2 = tf.sigmoid(i_64_2)
            f_64_2 = tf.sigmoid(f_64_2)
            g_64_2 = tf.nn.tanh(g_64_2)
            o_64_2 = tf.sigmoid(o_64_2)
            c_64_2 = f_64_2 * c_64_1 + i_64_2 * g_64_2
            h_64_2 = o_64_2 * tf.nn.tanh(c_64_2)
            conv64_11 = tf.concat(axis = 3, values = [conv64_10, h_64_2])
            
            conv64_12 = Conv3(conv64_11, 16, stride = 1, name = '64patch_conv7')
            conv64_output2 = LeakyReLU(conv64_12)
#            conv64_output2 = self.REM1(conv64_13, name = '64patch_conv8')
            
#            conv64_14 = tf.add(C3_2, conv64_output2)
            conv64_14 = conv64_2 + conv64_output2
            conv64_15 = Conv3(conv64_14, 16, stride = 1, name = '64patch_conv9')
            conv64_16 = LeakyReLU(conv64_15)
            conv64_17 = self.REM1(conv64_16, name = '64patch_conv10')
            
            i_64_3 = tf.layers.conv2d(conv64_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_64_3 = tf.layers.conv2d(conv64_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_64_3 = tf.layers.conv2d(conv64_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_64_3 = tf.layers.conv2d(conv64_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                      kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_64_3 = tf.sigmoid(i_64_3)
            f_64_3 = tf.sigmoid(f_64_3)
            g_64_3 = tf.nn.tanh(g_64_3)
            o_64_3 = tf.sigmoid(o_64_3)
            c_64_3 = f_64_3 * c_64_2 + i_64_3 * g_64_3
            h_64_3 = o_64_3 * tf.nn.tanh(c_64_3)
            conv64_18 = tf.concat(axis = 3, values = [conv64_17, h_64_3])
            
            conv64_19 = Conv3(conv64_18, 16, stride = 1, name = '64patch_conv11')
            conv64_output3 = LeakyReLU(conv64_19)
#            conv64_output3 = self.REM1(conv64_20, name = '64patch_conv12')
            #融合
            conv32_output3_up = tf.image.resize_bilinear(conv32_output3, size=(64,64))
            fuse_64_1 = tf.concat(axis = 3, values = [conv32_output3_up, Conv3(T_in_4down, 16, stride = 1, name = 'tm3')])
#            fuse_64_1 = tf.concat(axis = 3, values = [conv64_output3, Conv3(T_in_4down, 16, stride = 1, name = 'tm3')])
            fuse_64_2 = self.SE(fuse_64_1, out_dim = 32, ratio = 16, layer_name = 'fusion_64')
            conv64_21 = Conv3(fuse_64_2, 16, stride = 1, name = '64patch_conv13')
            conv64_22 = LeakyReLU(conv64_21)
#            conv64_output4 = tf.add(conv64_output3, conv64_22)
            conv64_output4 = conv64_output3 + conv64_22
            
            #128*128 patch
            conv128_1 = Conv3(C2, 16, stride = 1, name = '128patch_conv1')
            conv128_2 = LeakyReLU(conv128_1)
            conv128_3 = self.REM1(conv128_2, name = '128patch_conv2')
            
            i_128_1 = tf.layers.conv2d(conv128_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_128_1 = tf.layers.conv2d(conv128_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_128_1 = tf.layers.conv2d(conv128_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_128_1 = tf.layers.conv2d(conv128_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_128_1 = tf.sigmoid(i_128_1)
            f_128_1 = tf.sigmoid(f_128_1)
            g_128_1 = tf.nn.tanh(g_128_1)
            o_128_1 = tf.sigmoid(o_128_1)
            c_128_1 = i_128_1 * g_128_1
            h_128_1 = o_128_1 * tf.nn.tanh(c_128_1)
            conv128_4 = tf.concat(axis = 3, values = [conv128_3, h_128_1])
            
            conv128_5 = Conv3(conv128_4, 16, stride = 1, name = '128patch_conv3')
            conv128_output1 = LeakyReLU(conv128_5)
#            conv128_output1 = self.REM1(conv128_6, name = '128patch_conv4')
      
#            conv128_7 = tf.add(C2_2, conv128_output1)
            conv128_7 = conv128_2 + conv128_output1
            conv128_8 = Conv3(conv128_7, 16, stride = 1, name = '128patch_conv5')
            conv128_9 = LeakyReLU(conv128_8)
            conv128_10 = self.REM1(conv128_9, name = '128patch_conv6')
            
            i_128_2 = tf.layers.conv2d(conv128_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_128_2 = tf.layers.conv2d(conv128_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_128_2 = tf.layers.conv2d(conv128_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_128_2 = tf.layers.conv2d(conv128_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_128_2 = tf.sigmoid(i_128_2)
            f_128_2 = tf.sigmoid(f_128_2)
            g_128_2 = tf.nn.tanh(g_128_2)
            o_128_2 = tf.sigmoid(o_128_2)
            c_128_2 = f_128_2 * c_128_1 + i_128_2 * g_128_2
            h_128_2 = o_128_2 * tf.nn.tanh(c_128_2)
            conv128_11 = tf.concat(axis = 3, values = [conv128_10, h_128_2])
            
            conv128_12 = Conv3(conv128_11, 16, stride = 1, name = '128patch_conv7')
            conv128_output2 = LeakyReLU(conv128_12)
#            conv128_output2 = self.REM1(conv128_13, name = '128patch_conv8')
            
#            conv128_14 = tf.add(C2_2, conv128_output2)
            conv128_14 = conv128_2 + conv128_output2
            conv128_15 = Conv3(conv128_14, 16, stride = 1, name = '128patch_conv9')
            conv128_16 = LeakyReLU(conv128_15)
            conv128_17 = self.REM1(conv128_16, name = '128patch_conv10')
            
            i_128_3 = tf.layers.conv2d(conv128_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_128_3 = tf.layers.conv2d(conv128_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_128_3 = tf.layers.conv2d(conv128_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_128_3 = tf.layers.conv2d(conv128_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_128_3 = tf.sigmoid(i_128_3)
            f_128_3 = tf.sigmoid(f_128_3)
            g_128_3 = tf.nn.tanh(g_128_3)
            o_128_3 = tf.sigmoid(o_128_3)
            c_128_3 = f_128_3 * c_128_2 + i_128_3 * g_128_3
            h_128_3 = o_128_3 * tf.nn.tanh(c_128_3)
            conv128_18 = tf.concat(axis = 3, values = [conv128_17, h_128_3])
            
            conv128_19 = Conv3(conv128_18, 16, stride = 1, name = '128patch_conv11')
            conv128_output3 = LeakyReLU(conv128_19)
#            conv128_output3 = self.REM1(conv128_20, name = '128patch_conv12')
            
            conv64_output4_up = tf.image.resize_bilinear(conv64_output4, size=(128,128))
            fuse_128_1 = tf.concat(axis = 3, values = [conv64_output4_up, Conv3(T_in_2down, 16, stride = 1, name = 'tm2')])
#            fuse_128_1 = tf.concat(axis = 3, values = [conv128_output3, Conv3(T_in_2down, 16, stride = 1, name = 'tm2')])
            fuse_128_2 = self.SE(fuse_128_1, out_dim = 32, ratio = 16, layer_name = 'fusion_128')
            conv128_21 = Conv3(fuse_128_2, 16, stride = 1, name = '128patch_conv13')
            conv128_22 = LeakyReLU(conv128_21)
#            conv128_output4 = tf.add(conv128_output3, conv128_22)
            conv128_output4 = conv128_output3 + conv128_22
            
            #256*256 patch
            conv256_1 = Conv3(self.RealA, 16, stride = 1, name = '256patch_conv1')
            conv256_2 = LeakyReLU(conv256_1)
            conv256_3 = self.REM1(conv256_2, name = '256patch_conv2')
            
            i_256_1 = tf.layers.conv2d(conv256_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_256_1 = tf.layers.conv2d(conv256_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_256_1 = tf.layers.conv2d(conv256_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_256_1 = tf.layers.conv2d(conv256_3, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_256_1 = tf.sigmoid(i_256_1)
            f_256_1 = tf.sigmoid(f_256_1)
            g_256_1 = tf.nn.tanh(g_256_1)
            o_256_1 = tf.sigmoid(o_256_1)
            c_256_1 = i_256_1 * g_256_1
            h_256_1 = o_256_1 * tf.nn.tanh(c_256_1)
            conv256_4 = tf.concat(axis = 3, values = [conv256_3, h_256_1])
            
            conv256_5 = Conv3(conv256_4, 16, stride = 1, name = '256patch_conv3')
            conv256_output1 = LeakyReLU(conv256_5)
#            conv256_output1 = self.REM1(conv256_6, name = '256patch_conv4')
            
#            conv256_7 = tf.add(Conv3(self.RealA, 16, stride = 1, name = '256patch_1'), conv256_output1)
            conv256_7 = Conv3(self.RealA, 16, stride = 1, name = '256patch_1') + conv256_output1
            conv256_8 = Conv3(conv256_7, 16, stride = 1, name = '256patch_conv5')
            conv256_9 = LeakyReLU(conv256_8)
            conv256_10 = self.REM1(conv256_9, name = '256patch_conv6')
            
            i_256_2 = tf.layers.conv2d(conv256_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_256_2 = tf.layers.conv2d(conv256_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_256_2 = tf.layers.conv2d(conv256_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_256_2 = tf.layers.conv2d(conv256_10, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_256_2 = tf.sigmoid(i_256_2)
            f_256_2 = tf.sigmoid(f_256_2)
            g_256_2 = tf.nn.tanh(g_256_2)
            o_256_2 = tf.sigmoid(o_256_2)
            c_256_2 = f_256_2 * c_256_1 + i_256_2 * g_256_2
            h_256_2 = o_256_2 * tf.nn.tanh(c_256_2)
            conv256_11 = tf.concat(axis = 3, values = [conv256_10, h_256_2])
            
            conv256_12 = Conv3(conv256_11, 16, stride = 1, name = '256patch_conv7')
            conv256_output2 = LeakyReLU(conv256_12)
#            conv256_output2 = self.REM1(conv256_13, name = '256patch_conv8')
            
#            conv256_14 = tf.add(Conv3(self.RealA, 16, stride = 1, name = '256patch_2'), conv256_output2)
            conv256_14 = Conv3(self.RealA, 16, stride = 1, name = '256patch_2') + conv256_output2
            conv256_15 = Conv3(conv256_14, 16, stride = 1, name = '256patch_conv9')
            conv256_16 = LeakyReLU(conv256_15)
            conv256_17 = self.REM1(conv256_16, name = '256patch_conv10')
            
            i_256_3 = tf.layers.conv2d(conv256_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            f_256_3 = tf.layers.conv2d(conv256_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            g_256_3 = tf.layers.conv2d(conv256_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            o_256_3 = tf.layers.conv2d(conv256_17, filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu,
                                       kernel_initializer = tf.orthogonal_initializer(), bias_initializer = tf.constant_initializer(0),)
            i_256_3 = tf.sigmoid(i_256_3)
            f_256_3 = tf.sigmoid(f_256_3)
            g_256_3 = tf.nn.tanh(g_256_3)
            o_256_3 = tf.sigmoid(o_256_3)
            c_256_3 = f_256_3 * c_256_2 + i_256_3 * g_256_3
            h_256_3 = o_256_3 * tf.nn.tanh(c_256_3)
            conv256_18 = tf.concat(axis = 3, values = [conv256_17, h_256_3])
            
            conv256_19 = Conv3(conv256_18, 16, stride = 1, name = '256patch_conv11')
            conv256_output3 = LeakyReLU(conv256_19)
#            conv256_output3 = self.REM1(conv256_20, name = '256patch_conv12')
            
            conv128_output4_up = tf.image.resize_bilinear(conv128_output4, size=(256,256))
            fuse_256_1 = tf.concat(axis = 3, values = [conv128_output4_up, Conv3(self.images_wb, 16, stride = 1, name = 'tm1')])
            fuse_256_2 = self.SE(fuse_256_1, out_dim = 32, ratio = 16, layer_name = 'fusion_256')
            conv256_21 = Conv3(fuse_256_2, 16, stride = 1, name = '256patch_conv13')
            conv256_22 = LeakyReLU(conv256_21)
#            conv256_output4 = Conv3(tf.add(conv256_output3, conv256_22), 3, stride = 1, name = 'output')
            conv256_output4 = Conv3(conv256_output3 + conv256_22, 3, stride = 1, name = 'output')
            output = conv256_output4
            
        return tf.nn.tanh(output)


#判别网络
    def discriminator(self, input_, target, stride = 2, layer_count = 4):
        """
        Using the PatchGAN as a discriminator
        """
        input_ = tf.concat([input_, target], axis=3, name='Concat')
        layer_specs = self.num_discriminator_filters * np.array([1, 2, 4, 8])

        for i, output_channels in enumerate(layer_specs, 1):

            with tf.variable_scope('Layer' + str(i)):
         
                if i != 1:
                    input_ = BatchNorm(input_, isTrain = self.isTrain)
         
                if i == layer_count:
                    stride = 1
         
                input_ = LeakyReLU(input_)
                input_ = Conv(input_, output_channels = output_channels, kernel_size = 4, stride = stride, padding = 'VALID', mode = 'discriminator')

        with tf.variable_scope('Final_Layer',reuse=tf.AUTO_REUSE):
            output = Conv(input_, output_channels = 1, kernel_size = 4, stride = 1, padding = 'VALID', mode = 'discriminator')

        return tf.sigmoid(output)

    def build_vgg(self, img):

        model = vgg19.Vgg19()
        img = tf.image.resize_images(img, [224, 224])
        layer = model.feature_map(img)
        return layer
    
    
#  placeholder的作用理解为是占位符，形参就是占位置，用来代替实际参数，在实际调用该方法的时候传入实参。
#  输入参数共有三个：dtype，shape，name    dtype：表示输入的张量数据类型，常用的有float32，int32，float64等
#  shape：表示输入的张量大小，默认是None，也可以表示多维，如 (2, 3) 表示2行3列，(None, 4) 表示4列但行数不确定
#  name：表示输入张量的名称
    def build_model(self):

        with tf.variable_scope('Placeholders',reuse=tf.AUTO_REUSE):
            self.RealA = tf.placeholder(name='A', shape=[None, 256, 256, 3], dtype=tf.float32)
            self.RealB = tf.placeholder(name='B', shape=[None, 256, 256, 3], dtype=tf.float32)
            self.images_wb = tf.placeholder(name='images_wb', shape=[None, 256, 256, 3], dtype=tf.float32)
            #self.images_gc = tf.placeholder(name='images_gc', shape=[None, 256, 256, 3], dtype=tf.float32)   
            self.isTrain = tf.placeholder(name = "isTrain", shape = None, dtype = tf.bool)
            self.step = tf.train.get_or_create_global_step()

# tf.variable_scope()的作用都是为了不传引用而访问跨代码区域变量的一种方式，其内部功能是在其代码块内显式创建的变量
        with tf.variable_scope('Generator',reuse=tf.AUTO_REUSE):
            self.FakeB = self.generator()

        with tf.name_scope('Real_Discriminator'):
            with tf.variable_scope('Discriminator'):
                self.predict_real = self.discriminator(self.RealA, self.RealB)

        with tf.name_scope('Fake_Discriminator'):
            with tf.variable_scope('Discriminator', reuse=True):
                self.predict_fake = self.discriminator(self.RealA, self.FakeB)

        with tf.name_scope('Real_VGG'):
            with tf.variable_scope('VGG'):
                self.RealB_VGG = self.build_vgg(self.RealB)

        with tf.name_scope('Fake_VGG'):
            with tf.variable_scope('VGG', reuse=True):
                self.FakeB_VGG = self.build_vgg(self.FakeB)

        with tf.name_scope('DiscriminatorLoss'):
            self.D_loss = tf.reduce_mean(-(tf.log(self.predict_real + self.EPS) + tf.log(1 - self.predict_fake + self.EPS)))

        with tf.name_scope('GeneratorLoss'):
            self.gan_loss = tf.reduce_mean(-tf.log(self.predict_fake + self.EPS))
            self.l1_loss = tf.reduce_mean(tf.abs(self.RealB - self.FakeB))
            self.vgg_loss = (1e-5) * tf.losses.mean_squared_error(self.RealB_VGG, self.FakeB_VGG)
            self.G_loss = self.gan_wt * self.gan_loss + self.l1_wt * self.l1_loss + self.vgg_wt * self.vgg_loss
            #self.G_loss = self.gan_loss * self.gan_wt + self.vgg_wt * self.vgg_loss
        with tf.name_scope('Summary'):
            D_loss_sum = tf.summary.scalar('Discriminator Loss', self.D_loss)   # 添加变量
            G_loss_sum = tf.summary.scalar('Generator Loss', self.G_loss)
            gan_loss_sum = tf.summary.scalar('GAN Loss', self.gan_loss)
            l1_loss_sum = tf.summary.scalar('L1 Loss', self.l1_loss)
            vgg_loss_sum = tf.summary.scalar('VGG Loss', self.vgg_loss)
            '''output_img = tf.summary.image('Output', self.FakeB, max_outputs = 1)
            target_img = tf.summary.image('Target', self.RealB, max_outputs = 1)
            input_img = tf.summary.image('Input', self.RealA, max_outputs = 1)
            wb_img = tf.summary.image('WB', self.wb1, max_outputs = 1)
            #ce_img = tf.summary.image('CE', self.ce1, max_outputs = 1)
            gc_img = tf.summary.image('GC', self.gc1, max_outputs = 1)'''

            #self.image_summary = tf.summary.merge([output_img, target_img, input_img, wb_img, gc_img])
            self.G_summary = tf.summary.merge([gan_loss_sum, G_loss_sum, l1_loss_sum, vgg_loss_sum])
            # tf.summary.merge_all 可以将所有summary全部保存到磁盘
            self.D_summary = D_loss_sum

        with tf.name_scope('Variables'):
            self.G_vars = [var for var in tf.trainable_variables() if var.name.startswith("Generator")]
            self.D_vars = [var for var in tf.trainable_variables() if var.name.startswith("Discriminator")]

        with tf.name_scope('Save'):
            self.saver = tf.train.Saver()

        with tf.name_scope('Optimizer'):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):

                with tf.name_scope("Discriminator_Train"):
                    D_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
                    self.D_grads_and_vars = D_optimizer.compute_gradients(self.D_loss, var_list = self.D_vars)
                    self.D_train = D_optimizer.apply_gradients(self.D_grads_and_vars, global_step = self.step)

                with tf.name_scope("Generator_Train"):
                    G_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
                    self.G_grads_and_vars = G_optimizer.compute_gradients(self.G_loss, var_list = self.G_vars)
                    self.G_train = G_optimizer.apply_gradients(self.G_grads_and_vars, global_step = self.step)

    def train(self):

        logger_frequency = self.logger_frequency
        val_threshold = self.val_threshold

        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)

        print('Loading Model')
        self.build_model()
        print('Model Loaded')

        print('Loading Data')

        if self.custom_data:

            # Please ensure that the input images and target images have
            # the same filename.

            data = sorted(os.listdir(self.A_dir))
            total_image_count = int(len(data))   
            batches = total_image_count // self.batch_size
            train_data = data[: total_image_count]

            self.A_train = np.zeros((total_image_count, 256, 256, 3))
            self.B_train = np.zeros((total_image_count, 256, 256, 3))
            self.wb_train = np.zeros((total_image_count, 256, 256, 3))
            #self.ce_train = np.zeros((total_image_count, 256, 256, 3))
            #self.gc_train = np.zeros((total_image_count, 256, 256, 3))


            print(self.A_train.shape)

            for i, file in enumerate(train_data):
                self.A_train[i] = cv2.imread(os.path.join(os.getcwd(), self.A_dir, file), 1).astype(np.float32)
                self.B_train[i] = cv2.imread(os.path.join(os.getcwd(), self.B_dir, file), 1).astype(np.float32)
                self.wb_train[i] = cv2.imread(os.path.join(os.getcwd(), self.wb_dir, file), 1).astype(np.float32)
                #self.ce_train[i] = cv2.imread(os.path.join(os.getcwd(), self.ce_dir, file), 1).astype(np.float32)
                #self.gc_train[i] = cv2.imread(os.path.join(os.getcwd(), self.gc_dir, file), 1).astype(np.float32)

        else:
    
            self.A_train = np.load('A_train.npy').astype(np.float32)
            self.B_train = np.load('B_train.npy').astype(np.float32)
            

            total_image_count = len(self.A_train)
            batches = total_image_count // self.batch_size

        self.A_train = (self.A_train / 255) * 2 - 1
        self.B_train = (self.B_train / 255) * 2 - 1
        self.wb_train = (self.wb_train / 255) * 2 - 1
        #self.ce_train = (self.ce_train / 255) * 2 - 1
        #self.gc_train = (self.gc_train / 255) * 2 - 1
    
        print('Data Loaded')
        

        with tf.Session() as self.sess:

            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            if self.restore:
                print('Loading Checkpoint')
                ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
                self.saver.restore(self.sess, ckpt)
                self.step = tf.train.get_or_create_global_step()
                print('Checkpoint Loaded')

            self.writer = tf.summary.FileWriter(self.tensorboard_dir, tf.get_default_graph())
            total_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
            G_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables() if v.name.startswith("Generator")])
            D_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables() if v.name.startswith("Discriminator")])
            loss_operations = [self.D_loss, self.G_loss, self.gan_loss, self.l1_loss, self.vgg_loss]


            counts = self.sess.run([G_parameter_count, D_parameter_count, total_parameter_count])

            print('Generator parameter count:', counts[0])
            print('Discriminator parameter count:', counts[1])
            print('Total parameter count:', counts[2])

            # The variable below is divided by 2 since both the Generator 
            # and the Discriminator increases step count by 1
            start = self.step.eval() // (batches * 2)

            for i in range(start, self.epochs):

                print('Epoch:', i)
                shuffle = np.random.permutation(total_image_count)

                for j in range(batches):

                    if j != batches - 1:
                        current_batch = shuffle[j * self.batch_size: (j + 1) * self.batch_size]
                    else:
                        current_batch = shuffle[j * self.batch_size: ]

                    a = self.A_train[current_batch]
                    b = self.B_train[current_batch]
                    wb = self.wb_train[current_batch]
                    #ce = self.ce_train[current_batch]
                    #gc = self.gc_train[current_batch]
                    
                    feed_dict = {self.RealA: a, self.RealB: b, self.images_wb: wb, self.isTrain: True}

                    begin = time.time()
                    step = self.step.eval()

                    _, D_summary = self.sess.run([self.D_train, self.D_summary], feed_dict = feed_dict)

                    self.writer.add_summary(D_summary, step)

                    _, G_summary = self.sess.run([self.G_train, self.G_summary], feed_dict = feed_dict)

                    self.writer.add_summary(G_summary, step)

                    if j % logger_frequency == 0:
                        D_loss, G_loss, GAN_loss, L1_loss, VGG_loss= self.sess.run(loss_operations, feed_dict=feed_dict)

                        GAN_loss = GAN_loss * self.gan_wt
                        L1_loss = L1_loss * self.l1_wt
                        VGG_loss = VGG_loss * self.vgg_wt

                        trial_image_idx = np.random.randint(total_image_count)
                        a = self.A_train[trial_image_idx]
                        b = self.B_train[trial_image_idx]

                        if a.ndim == 3:
                            a = np.expand_dims(a, axis = 0)

                        if b.ndim == 3:
                            b = np.expand_dims(b, axis = 0)
                            
                        if wb.ndim == 3:
                            wb = np.expand_dims(wb, axis = 0)
                            
                        '''if ce.ndim == 3:
                            ce = np.expand_dims(ce, axis = 0)'''

                        '''if gc.ndim == 3:
                            gc = np.expand_dims(gc, axis = 0)'''   

                        '''feed_dict = {self.RealA: a, self.RealB: b, self.images_wb: wb, self.images_gc: gc, self.isTrain: False}
                        img_summary = self.sess.run(self.image_summary, feed_dict=feed_dict)
                        self.writer.add_summary(img_summary, step)'''

                    G_D_step = step // 2
                    if step % 100 == 0:  

                        print('step   :{}'.format(step))
                        print('D_Loss  :{}'.format(D_loss))
                        print('G_Loss  :{}'.format(G_loss))
                        print('GAN_Loss :{}'.format(GAN_loss))
                        print('L1_Loss  :{}'.format(L1_loss))
                        print('VGG_Loss :{}'.format(VGG_loss))
                        '''line = 'Batch: %d, D_Loss: %.3f, G_Loss: %.3f, GAN: %.3f, L1: %.3f, P: %.3f' % (
                            j, D_loss, G_loss, GAN_loss, L1_loss, VGG_loss)
                        print (line)  '''
     
                    if step % 1000 == 0:
                        print('GD', G_D_step, 'val', val_threshold)
                        save_path = self.saver.save(self.sess, os.path.join(self.ckpt_dir, 'gan'), global_step = self.step.eval())
                        print('Model saved in file:%s' % save_path)


    def inference(self, input_dir, wb, result_dir):

        input_list = os.listdir(input_dir)
    
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        print('Loading Model')
        self.build_model()
        print('Model Loaded')

        with tf.Session() as self.sess:

            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            print('Loading Checkpoint')
            ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
            self.saver.restore(self.sess, ckpt)
            self.step = tf.train.get_or_create_global_step()
            print('Checkpoint Loaded')
            time_start = time.time()  # 开始计时

            for i, img_file in enumerate(input_list, 1):

                RealA1 = cv2.imread(os.path.join(input_dir, img_file), 1)
                wb1 = cv2.imread(os.path.join(wb, img_file), 1)
                #ce1 = cv2.imread(os.path.join(ce, img_file), 1)
                #gc1 = cv2.imread(os.path.join(gc, img_file), 1)
                
                RealA1 = ((np.expand_dims(RealA1, axis = 0) / 255) * 2) - 1
                wb1 = ((np.expand_dims(wb1, axis = 0) / 255) * 2) - 1
                #ce1 = ((np.expand_dims(ce1, axis = 0) / 255) * 2) - 1
                #gc1 = ((np.expand_dims(gc1, axis = 0) / 255) * 2) - 1
                print('Processing image', i)

                feed_dict = {self.RealA: RealA1, self.images_wb: wb1, self.isTrain: False}
                generated_B = self.FakeB.eval(feed_dict = feed_dict)
                generated_B = (((generated_B[0] + 1)/2) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(result_dir, img_file), generated_B)
                
#                graph = tf.get_default_graph()
#                flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
#                params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
#                print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))  
            print('Done.')
            time_end = time.time()  # 结束计时
            time_c = time_end - time_start  # 运行所花时间
            print('time cost', time_c, 's')
