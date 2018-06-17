import os 
import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt 
import numpy as np
import cv2

def get_file(file_dir):
    images = []
    temp = []
    for root,sub_folders,files in os.walk(file_dir):  #返回根目录，子目录，子目录下的文件
        # image directories
        for name in files:
            images.append(os.path.join(root,name))
        # get 2 sub_folders name
        for name in sub_folders:
            temp.append(os.path.join(root,name))

    # assign 10 labels based on the folder names
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]

        if letter =='0':
            labels = np.append(labels,n_img*[0])
        elif letter =='1':
            labels = np.append(labels,n_img*[1])
        elif letter =='2':
            labels = np.append(labels,n_img*[2])
        elif letter =='3':
            labels = np.append(labels,n_img*[3])
        elif letter =='4':
            labels = np.append(labels,n_img*[4])
        elif letter =='5':
            labels = np.append(labels,n_img*[5])
        elif letter =='6':
            labels = np.append(labels,n_img*[6])
        elif letter =='7':
            labels = np.append(labels,n_img*[7])
        elif letter =='8':
            labels = np.append(labels,n_img*[8])
        else:
            labels = np.append(labels,n_img*[9])

     # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)  #打乱次序

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]

    return image_list,label_list

def set_tfrecords():
    file_name = './testimage/pic2/'
    images,labels=get_file(file_name)
    n_samples = len(labels)
    if np.shape(images)[0] != n_samples:
        raise ValueError('Image size %d does not match label size %d.'%(images.shape[0],n_samples))
    
    writer= tf.python_io.TFRecordWriter("train.tfrecords") #要生成的文件
    print('\nTransform start....')
    for i in np.arange(0,n_samples):
        try:
            img=cv2.imread(images[i])
            img_raw=img.tobytes()#将图片转化为二进制格式
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
               'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
               'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
              })) #example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  #序列化为字符串
        except IOError as e:
            print('Could not read:',images[i])
            print('error:%s'%e)
            print('skip it!\n')
 
    writer.close()
    print("Transform done!")

def read_tfrecords(filename): # 读入tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 3])  #reshape为28*28的3通道图片
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    return img, label

def get_batch(image, label, batch_size):
    images, label_batch = tf.train.shuffle_batch([image, label],batch_size=batch_size,num_threads=8,
                                                 capacity=2000,min_after_dequeue=1000)
    
    return images,tf.reshape(label_batch, [batch_size])

def get_Tst_batch(image, label, batch_size):
     images, label_batch=tf.train.batch([image, label],batch_size=batch_size,num_threads=8,capacity=2000)
     return images,tf.reshape(label_batch, [batch_size])
    


def check_batch():
    #set_tfrecords()
    image,label=read_tfrecords("train.tfrecords")
    batch_image,batch_label=get_batch(image,label,5)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i<1:
                img,label=sess.run([batch_image,batch_label])
                for j in range(5):
                    print("label: %d"%label[j])
                    plt.imshow(img[j])
                    plt.show()
                    
                i+=1
        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
        coord.join(threads) 
   

if __name__=='__main__':
    check_batch()
    
