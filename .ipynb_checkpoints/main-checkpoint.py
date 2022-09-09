import os
import argparse
from model import GAN
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()   #创建解析器 '''名字，type参数类型，default默认值，help:参数描述，metavar'''

parser.add_argument("--lr", help="Learning Rate (Default = 0.001)", #学习率
                    type = float, default = 0.0005)
parser.add_argument("--D_filters", help="Number of filters in the 1st conv layer of the discriminator (Default = 64)",
                    type = int, default = 64)   #鉴别器的第一个conv层中的筛选器数
parser.add_argument("--layers", help="Number of layers per dense block (Default = 4)",   #每个密集块的层数
                    type = int, default = 2)
parser.add_argument("--growth_rate", help="Growth Rate of the dense block (Default = 12) ",   #密集块生长速率
                    type = int, default = 12)
parser.add_argument("--gan_wt", help="Weight of the GAN loss factor (Default = 2)",
                    type = float, default = 2)
parser.add_argument("--l1_wt", help="Weight of the L1 loss factor (Default = 100)",
                    type = float, default = 60)
parser.add_argument("--vgg_wt", help="Weight of the VGG loss factor (Default = 10)",
                    type = float, default = 5)
parser.add_argument("--restore", help = "Restore checkpoint for training (Default = False)",  #恢复训练检查点
                    type = bool, default = False)
parser.add_argument("--batch_size", help="Set the batch size (Default = 1)",   #设置批量大小  一次性读入几张图片 带不起来改为2
                    type = int, default = 2)
parser.add_argument("--decay", help="Batchnorm decay (Default = 0.99)",    #批次范数衰减
                    type = float, default = 0.99)
parser.add_argument("--epochs", help = "Epochs (Default = 200)", #数据集全部训练一次叫一个epoch 可修改次数
                    type = int, default = 20)
parser.add_argument("--model_name", help = "Set a model name",
                    default = 'model')
parser.add_argument("--save_samples", help = "Generate image samples after validation (Default = False)",
                    type = bool, default = True) 
parser.add_argument("--sample_image_dir", help = "Directory containing sample images (Used only if save_samples is True; Default = samples)",
                    default = 'samples')
parser.add_argument("--A_dir", help = "Directory containing the input images for training, testing or inference (Default = A)",
                    default = 'A')
parser.add_argument("--B_dir", help = "Directory containing the target images for training or testing. In inference mode, this is used to store results (Default = B)",
                    default = 'B')
parser.add_argument("--wb_dir", help = "Directory containing the input images for training, testing or inference (Default = A)",
                    default = 'WB')
#parser.add_argument("--ce_dir", help = "Directory containing the input images for training, testing or inference (Default = A)",
                    #default = 'HE')
#parser.add_argument("--gc_dir", help = "Directory containing the input images for training, testing or inference (Default = A)",
                    #default = 'GC')
parser.add_argument("--A1_dir", help = "Directory containing the input images for test",
                    default = 'Finally2/A')
#parser.add_argument("--gc1_dir", help = "Directory containing the input images for test",
                    #default = 'Finally2/GC')
parser.add_argument("--wb1_dir", help = "Directory containing the input images for test",
                    default = 'Finally2/WB')
parser.add_argument("--Result", help = "Directory containing the result images",
                    default = 'Test')
parser.add_argument("--custom_data", help = "Using your own data as input and target (Default = True)",
                    type = bool, default = True)
parser.add_argument("--val_fraction", help = "Fraction of dataset to be split for validation (Default = 0.15)",
                   type = float, default = 0.15)
parser.add_argument("--val_threshold", help = "Number of steps to wait before validation is enabled. (Default = 0)",
                   type = int, default = 0)
parser.add_argument("--logger_frequency", help = "Number of batches to wait before logging the next set of loss values (Default = 20)",
                   type = int, default = 5)
parser.add_argument("--mode", help = "Select between train, test or inference modes",
                    default = 'inference', choices = ['train', 'inference'])
if __name__ == '__main__':
   
    #jiazaiGPU 限制GPU用量
    config = tf.ConfigProto()   #tf.ConfigProto()函数用在创建session的时候，用来对session进行参数配置
    config.gpu_options.allow_growth = True ##不全部占满显存, 按需分配 
    sess = tf.Session(config=config)   #在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量
                                       #的赋值和计算都要通过tf.Session的run来进行
    KTF.set_session(sess)
 
    
    args = parser.parse_args()
    net = GAN(args)
    if args.mode == 'train':   #mode 模式 
        net.train() 
    if args.mode == 'inference':  #inference 推理
        net.inference(args.A1_dir, args.wb1_dir, args.Result)
        
