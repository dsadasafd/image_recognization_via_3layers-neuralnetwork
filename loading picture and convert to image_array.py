#!/usr/bin/env python
# coding: utf-8

# In[68]:


# python notebook for my own neural network
# code for a 3-layer neural network, and code for learning the mnist_dataset
# this version trains using the mnist_dataset_trains.csv, then tests on my written number image 28*28
# by zengyue


# In[69]:


import numpy
import matplotlib.pyplot
import scipy.special
get_ipython().run_line_magic('matplotlib', 'inline')


# In[70]:


import glob
import imageio


# In[71]:


# neural network class defination
class neuralNetwork:
    # initialise the network 
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0, pow(self.inodes,-0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    # train the network
    def train(self, inputs_list, targets_list):
        # convert inputlist to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # calculate the signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate the signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final outputs layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is (targets - outputs)
        output_errors = targets - final_outputs
        # hidden layer error 
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the who
        self.who += self.lr*numpy.dot(output_errors*final_outputs*(1.0-final_outputs), hidden_outputs.T)
        # update the wio 
        self.wih += self.lr*numpy.dot(hidden_errors*hidden_outputs*(1.0-hidden_outputs), inputs.T)
        pass
    
    # query the network
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    pass


# In[72]:


input_nodes = 784
hidden_nodes = 100
output_nodes = 10 
learning_rate = 0.1

my_own_neuralnetwork = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# In[73]:


# train the neural network using by mnist_dataset.csv
# load the training data 

training_data_file = open('C:\\Users\\Administrator\\Desktop\\mnist_dataset\\mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close

# start to train my own neural network
epochs = 10 

for e in range(epochs):
    
    for records in training_data_list:
        all_values = records.split(",")
        inputs = numpy.asfarray(all_values[1:])/255*0.99+0.01
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        my_own_neuralnetwork.train(inputs, targets)
        pass


# In[74]:


# test my written number image
# prepare our image data

our_own_image_data = []

for image_file_name in glob.glob('C:\\Users\\Administrator\\Desktop\\mnist_dataset\\my_own_image_?.png'):
    # use the filename to set the correct number
    label = int(image_file_name[-5:-4])
    print('loading the', image_file_name, 'and the number is ', label)
    
    image_array = imageio.v2.imread(image_file_name, as_gray=True)
    image_data = 255-image_array.reshape(784)
    image_data = image_data/255*0.99+0.01
    print(numpy.min(image_data))
    print(numpy.max(image_data))
    record = numpy.append(label, image_data)
    our_own_image_data.append(record)
    print(our_own_image_data)
    pass


# In[75]:


# start to test 
# plot the image 

matplotlib.pyplot.imshow(our_own_image_data[0][1:].reshape(28,28), cmap='Greys', interpolation='None')

correct_number = our_own_image_data[0][0]
inputs = our_own_image_data[0][1:]
outputs = my_own_neuralnetwork.query(inputs)
print(outputs)

test_results_numbers = numpy.argmax(outputs)
print('my neural network said the number in the photo is ', test_results_numbers)

if test_results_numbers == correct_number:
    print('congratulation, my own neural network work!')
else:
    print('fuck, damn it!')


# In[ ]:




