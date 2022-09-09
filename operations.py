import tensorflow as tf

def Conv7(input_, k, stride, name = None):
     
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    weights = _weights("weights_7",
      shape=[7, 7, input_.get_shape()[3], k])

    padded = tf.pad(input_, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
    conv = tf.nn.conv2d(padded, weights,
        strides=[1, stride, stride, 1], padding='VALID')

    return conv
   
def Conv3(input_, k, stride, name = None):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = _weights("weights_3",
            shape=[3, 3, input_.get_shape()[3], k])

        conv = tf.nn.conv2d(input_, weights,
            strides=[1, stride, stride, 1], padding='SAME')
        
        return conv
    
def Conv5(input_, k, stride,name = None):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = _weights("weights_5",
            shape=[3, 3, input_.get_shape()[3], k])

        conv = tf.nn.conv2d(input_, weights,
            strides=[1, stride, stride, 1], padding='SAME')
        
        return conv

def Conv1(input_, k, stride, name = None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = _weights("weights_1",
            shape=[1, 1, input_.get_shape()[3], k])

        conv = tf.nn.conv2d(input_, weights,
            strides=[1, stride, stride, 1], padding='SAME')
        
        return conv

    
def Conv(input_, kernel_size, stride, output_channels, padding = 'SAME', mode = None):

    with tf.variable_scope("Conv",reuse=tf.AUTO_REUSE):

        input_channels = input_.get_shape()[-1]
        kernel_shape = [kernel_size, kernel_size, input_channels, output_channels]

        kernel = tf.get_variable("Filter", shape = kernel_shape, dtype = tf.float32, initializer = tf.random_normal_initializer(mean=0.0,stddev=0.02,dtype=tf.float32))
        
        # Patchwise Discriminator (PatchGAN) requires some modifications.
        if mode == 'discriminator':
            input_ = tf.pad(input_, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="CONSTANT")

        return tf.nn.conv2d(input_, kernel, strides = [1, stride, stride, 1], padding = padding)
        
def TransposeConv(input_, output_channels, kernel_size = 4):

    with tf.variable_scope("TransposeConv",reuse=tf.AUTO_REUSE):

        input_height, input_width, input_channels = [int(d) for d in input_.get_shape()[1:]]
        batch_size = tf.shape(input_)[0] 

        kernel_shape = [kernel_size, kernel_size, output_channels, input_channels]
        output_shape = tf.stack([batch_size, input_height*2, input_width*2, output_channels])

        kernel = tf.get_variable(name = "filter", shape = kernel_shape, dtype=tf.float32, initializer = tf.random_normal_initializer(mean=0.0,stddev=0.02,dtype=tf.float32))
        
        return tf.nn.conv2d_transpose(input_, kernel, output_shape, [1, 2, 2, 1], padding="SAME")


def _weights(name, shape, mean=0.0, stddev=0.02):
#gussian
  var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
  return var

def _biases(name, shape, constant=0.0):

  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))

#def MaxPool(input_):
#    with tf.variable_scope("MaxPool"):
#        return tf.nn.max_pool(input_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def AvgPool(input_, k = 2, name="Avg"):
    with tf.variable_scope(name):
        return tf.nn.avg_pool(input_, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

def ReLU(input_,name = "Relu"):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return tf.nn.relu(input_)

def LeakyReLU(input_, leak = 0.2):
    with tf.variable_scope("LeakyReLU",reuse=tf.AUTO_REUSE):
        return tf.maximum(input_, leak * input_)

def BatchNorm(input_, isTrain, name='BN', decay = 0.99):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        return tf.contrib.layers.batch_norm(input_, is_training = isTrain, decay = decay)

def DropOut(input_, isTrain, rate=0.2, name='drop') :
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        return tf.layers.dropout(inputs=input_, rate=rate, training=isTrain)

def Global_Average_Pooling(x):
    return tf.keras.layers.GlobalAveragePooling2D()(x)

def Fully_connected(x, units=17, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)