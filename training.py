import os
import numpy as np
import tensorflow as tf
import input_data
import model
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math

N_CLASSES = 10
IMG_W = 28
IMG_H = 28
BATCH_SIZE = 50
CAPACITY = 2000
MAX_STEP = 10000   #至少要在10K以上
learning_rate = 0.001
LOGDIR = './log/'
def run_training():

    train_image,train_label=input_data.read_tfrecords('train.tfrecords')
    train_batch,train_label_batch = input_data.get_batch(train_image,train_label,BATCH_SIZE)
    
    train_logits = model.inference(train_batch,BATCH_SIZE,N_CLASSES,1)
    train_loss = model.softmax_loss(train_logits,train_label_batch)
    train_op = model.training(train_loss,learning_rate)
    train_acc = model.evaluation(train_logits,train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op,train_loss,train_acc])
            if step%50==0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%'%(step,tra_loss,tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)

            if step%2000==0:
                 checkpoint_path = os.path.join(LOGDIR,'model.ckpt')
                 saver.save(sess,checkpoint_path,global_step=step)
                 
    except tf.errors.OutOfRangeError:        
          print('Done training --epoch limit reached')
    finally:
             coord.request_stop()

    coord.join(threads)
    sess.close()

def get_one_image(train):

    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
    
    #image = Image.open(img_dir)
    image = cv2.imread(img_dir)
    plt.imshow(image)
    plt.show()
    #image = image.resize([28, 28])
    #image = tf.reshape(image,[1,28,28,3])
    image = np.array(image)
    return image

def evaluate_one_image():

    #test_dir = './testimage/pic2/'
    #test, test_label = input_data.get_file(test_dir)
    #image_array = get_one_image(test)

    image = plt.imread("test_2.jpg")
    plt.imshow(image)
    plt.show()
    # image_array = np.array(image)
    
    #with tf.Graph().as_default():
    BATCH_SIZE = 1
    N_CLASSES = 10
    
    x = tf.placeholder(tf.float32, shape=[28, 28, 3])   
    logit = model.inference(x, BATCH_SIZE, N_CLASSES,0)
    logits = tf.nn.softmax(logit)

    log_dir = './log'

    saver = tf.train.Saver()

    #with tf.Session() as sess:
    sess = tf.Session()
    print("Reading checkpoints....")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('Loading success,global_step is %s'%global_step)
    else:
        print('No checkpoint file found')

    prediction = sess.run(logit,feed_dict={x:image})
    prediction1 = sess.run(logits,feed_dict={x:image})
    print(prediction)
    print('\n')
    print(prediction1)
    max_index = np.argmax(prediction)
    if max_index ==0:
        print('This is 0 with possibility %.6f'%prediction1[:,0])
    elif max_index ==1:
        print('This is 1 with possibility %.6f'%prediction1[:,1])
    elif max_index ==2:
        print('This is 2 with possibility %.6f'%prediction1[:,2])
    elif max_index ==3:
        print('This is 3 with possibility %.6f'%prediction1[:,3])
    elif max_index ==4:
        print('This is 4 with possibility %.6f'%prediction1[:,4])
    elif max_index ==5:
        print('This is 5 with possibility %.6f'%prediction[:,5])
    elif max_index ==6:
        print('This is 6 with possibility %.6f'%prediction[:,6])
    elif max_index ==7:
        print('This is 7 with possibility %.6f'%prediction[:,7])
    elif max_index ==8:
        print('This is 8 with possibility %.6f'%prediction[:,8])
    else:       
        print('This is 9 with possibility %.6f'%prediction[:,9])

    sess.close() 

def check_batch():
    
    image,label=input_data.read_tfrecords('test.tfrecords')  #从训练集读取数据
    test_image,test_label = input_data.get_Tst_batch(image,label,100)

    test_logits = model.inference(test_image,100,10,0)
    test_correct = tf.nn.in_top_k(test_logits,test_label,1)
    #correct = tf.cast(test_correct,tf.float16)
    #test_accuracy = tf.reduce_mean(correct)

    n_test = 10000
    
    log_dir = './log'
    saver = tf.train.Saver(tf.global_variables())
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    print("Reading checkpoints....")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('Loading success,global_step is %s'%global_step)
    else:
        print('No checkpoint file found')
    try:
        num_iter = int(math.ceil(n_test/100))
        true_count = 0
        total_sample_count = num_iter*100
        step = 0
        while step < num_iter and not coord.should_stop(): 
               correct = sess.run(test_correct)
               true_count+=np.sum(correct)
               step+=1
               accuracy = true_count/total_sample_count
        print('test accuracy = %.3f' % accuracy)
    except tf.errors.OutOfRangeError:
            print('Done testing -- epoch limit reached')
              
    finally:
              coord.request_stop()
    coord.join(threads)
    sess.close()

def show_feature_map():
    image = plt.imread('test_2.jpg')
    plt.imshow(image)
    image = tf.cast(image,tf.float32)
    x = tf.reshape(image,[1,28,28,3])
    log_dir = './log'

    with tf.variable_scope('conv1'):
        w = tf.get_variable('weights',[5,5,3,32])
        b = tf.get_variable('biases',[32])
        conv1 = tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
        x_b1 = tf.nn.bias_add(conv1,b)
        x_relu1 = tf.nn.relu(x_b1)
    # with tf.variable_scope('pooling1') as scope:
    #     pool1 = tf.nn.max_pool(x_relu1,ksize=[1,2,2,1],strides=[1,2,2,1],
    #                            padding='SAME',name='pooling1')
    # with tf.variable_scope('conv2'):
    #     w = tf.get_variable('weights',[5,5,32,64])
    #     b = tf.get_variable('biases',[64])
    #     conv2 = tf.nn.conv2d(pool1,w,strides=[1,1,1,1],padding='SAME')
    #     x_b2 = tf.nn.bias_add(conv2,b)
    #     x_relu2 = tf.nn.relu(x_b2)
    n_feature = int(x_relu1.get_shape()[-1])
    print(n_feature,x_relu1.get_shape())
    saver = tf.train.Saver(tf.global_variables())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("Reading checkpoints....")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success,global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
    feature_map = tf.reshape(x_relu1,[28,28,n_feature])
    images = sess.run(feature_map)

    plt.figure(figsize=(10, 10))
    for i in np.arange(0, n_feature):
        plt.subplot(8, 4, i + 1)
        plt.axis('off')
        plt.imshow(images[:, :, i])
    plt.show()

if __name__=='__main__':
    evaluate_one_image()


























