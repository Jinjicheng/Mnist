import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def inference(images,batch_size,n_classes,flag):

    images=tf.reshape(images,shape=[-1,28,28,3])  #[batch,image_height,image_width,image_channels]
    images=(tf.cast(images,tf.float32)/255.-0.5)*2  #归一化处理
    # conv1
    with tf.variable_scope('conv1') as scope:
        #weights = tf.get_variable('weights',
        #                          shape = [5,5,3,32],
        #                         dtype = tf.float32,
        #                          initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32),)
        #biases = tf.get_variable('biases',
        #                         shape = [32],
        #                        dtype = tf.float32,
        #                         initializer=tf.constant_initializer(0.1))
        weights = tf.Variable(tf.truncated_normal([5,5,3,32],stddev=0.1),name='weights')
        biases = tf.Variable(tf.constant(0.1,shape=[32]),name='biases')
        conv = tf.nn.conv2d(images,weights,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name=scope.name)
        tf.summary.histogram("weights",weights)
        tf.summary.histogram("biases",biases)
        tf.summary.histogram("activation",conv1)

    # pool1
    with tf.variable_scope('pooling1') as scope:
        pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='SAME',name='pooling1')

    # conv2
    with tf.variable_scope('conv2') as scope:
          #weights = tf.get_variable('weights',
          #                        shape = [5,5,32,64],
          #                        dtype = tf.float32,
          #                        initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
          #biases = tf.get_variable('biases',
          #                       shape = [64],
          #                       dtype =tf.float32,
          #                       initializer=tf.constant_initializer(0.1))
          weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='weights')
          biases = tf.Variable(tf.constant(0.1, shape=[64]), name='biases')
          conv = tf.nn.conv2d(pool1,weights,strides=[1,1,1,1],padding='SAME')
          pre_activation = tf.nn.bias_add(conv,biases)
          conv2 = tf.nn.relu(pre_activation,name='conv2')
          tf.summary.histogram("weights", weights)
          tf.summary.histogram("biases", biases)
          tf.summary.histogram("activation", conv2)

    # pool2
    with tf.variable_scope('pooling2') as scope:
          pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='SAME',name='pooling2')

    # local3
    with tf.variable_scope('local3') as scope:
          reshape = tf.reshape(pool2,shape=[batch_size, 7*7*64])
          #dim = reshape.get_shape()[1].value
          #weights = tf.get_variable('weights',
          #                        shape = [7*7*64,1024],
          #                        dtype = tf.float32,
          #                        initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
          #biases = tf.get_variable('biases',
          #                       shape=[1024],
          #                       dtype=tf.float32,
          #                       initializer=tf.constant_initializer(0.1))
          weights = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1), name='weights')
          biases = tf.Variable(tf.constant(0.1, shape=[1024]), name='biases')
          if flag==1:
              drop = tf.nn.dropout(tf.matmul(reshape,weights)+biases,0.5)
          else:
              drop = tf.nn.dropout(tf.matmul(reshape,weights)+biases,1)
          local3 = tf.nn.relu(drop,name='local3')
          tf.summary.histogram("weights", weights)
          tf.summary.histogram("biases", biases)
          tf.summary.histogram("activation", local3)

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
          #weights = tf.get_variable('softmax_linear',
          #                        shape=[1024,n_classes],
          #                        dtype=tf.float32,
          #                        initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
          #biases = tf.get_variable('biases',
          #                        shape=[n_classes],
          #                        dtype=tf.float32,
          #                        initializer=tf.constant_initializer(0.1))
          weights = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='weights')
          biases = tf.Variable(tf.constant(0.1, shape=[10]), name='biases')
          softmax_linear = tf.add(tf.matmul(local3,weights),biases,name='softmax_linear')
          tf.summary.histogram("weights", weights)
          tf.summary.histogram("biases", biases)
          #tf.summary.histogram("activation", conv1)
    return softmax_linear

def softmax_loss(logits,labels):
    with tf.variable_scope('loss') as scope:
         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                         (logits=logits,labels=labels,name='xentropy_per_example')
         loss = tf.reduce_mean(cross_entropy,name='loss')
         tf.summary.scalar(scope.name+'/loss',loss)
    return loss

def training(loss,learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
        
    return train_op

def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
        
    return accuracy    

        


















        



      
