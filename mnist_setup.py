# @author Scott Dobbins
# @version 0.1
# @date 2017-12-02

### imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# removable
print(tf.__version__)

### setup
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

### get train, test, and validation
train_images = mnist.train.images.reshape((-1,28,28))
train_labels = mnist.train.labels
test_images = mnist.test.images.reshape((-1,28,28))
test_labels = mnist.test.labels
validation_images = mnist.validation.images.reshape((-1,28,28))
validation_labels = mnist.validation.labels

### set up investigation images
train = np.zeros((10,2) + train_images.shape[1:3])
train[:,0,:,:] = np.take(train_images, [7,4,16,1,2,27,3,0,5,8], axis = 0)
train[:,1,:,:] = np.take(train_images, [10,12,76,11,33,28,18,14,43,44], axis = 0)

