import os
import cv2
import h5py
import numpy as np
from skimage.transform import resize

if __name__ == '__main__':

    if not os.path.exists('A'):  #os.path.exists()判断括号里的文件是否存在，括号内的可以是文件路径
        os.mkdir('A')   #使用os.mkdir()函数创建目录

    if not os.path.exists('B'):
        os.mkdir('B')

    with h5py.File('data.mat', 'r') as f:
        images = np.array(f['images'])
        depths = np.array(f['depths'])

    images = images.transpose(0, 1, 3, 2)  #将第2行第3行进行互换
    depths = depths.transpose(2, 1, 0)
    depths = (depths - np.min(depths, axis = (0, 1))) / np.max(depths, axis = (0, 1))
    # np.min取最小值  axis = (0,1)遍历整个矩阵
    depths = ((1 - depths) * np.random.uniform(0.2, 0.4, size = (1449, ))).transpose(2, 0, 1)
#numpy.random.uniform(low,high,size) 从一个均匀分布[low,high)中随机采样，
#注意定义域是左闭右开，即包含low，不包含high.
    for i in range(len(images)):   #for i in range () 就是给i赋值
        fog = (images[i] * depths[i]) + (1 - depths[i]) * np.ones_like(depths[i]) * 255
        fog = resize(fog.transpose(1, 2, 0), (256, 256, 3), mode = 'reflect')
        img = resize(images[i].transpose(1, 2, 0), (256, 256, 3), mode = 'reflect')
        img = (img * 255).astype(np.uint8)

        cv2.imwrite(os.path.join('A', str(i).zfill(4) + '.png'), fog)
        cv2.imwrite(os.path.join('B', str(i).zfill(4) + '.png'), img)
        
        print('Extracting image:', i, end = '\r')

    print('Done.')
