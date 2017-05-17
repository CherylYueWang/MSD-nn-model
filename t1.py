'''
Copyright 2017 Yue Wang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''


import numpy as np
import tensorflow as tf
import random

def getShuffleBatch(dataset, labelset, size):
	# return a shuffled batch of size
	pass


def readData(filename, filetype):
	data = [] # list of list
	categories = [] # list
	with open(filename,'r') as file:
		firstline = file.readline()
		categories = firstline.split(',')
		allData = file.readlines()
		for line in allData:
			data.append(line.split(','))

	# extract all labels, arrange in a dictionary
	labelsTemp = []
	for row in data:
		if row[0] not in labelsTemp:
			labelsTemp.append(row[0])
	labels = {}
	i = 0
	for l in labelsTemp:
		if l not in labels.keys():
			labels[l] = i
			i+=1
	numLabels = len(labelsTemp)

	#print(labels)
	#print(categories)
	#print(data[1])
	#print(len(data))
	
	# create label matrix for training
	trainLabel = []
	# extract labels
	for row in data:
		index = labels[row[0]]
		label = [0]*numLabels # all other entries are 0's
		label[index] = 1 # only one entry is 1
		trainLabel.append(label)
	#print(trainLabel)

	print('got raw input rows',len(data))


	if filetype=='train':
		# delete NA entries
		for row in data:
			del row[0]
			del row[-1]
			del row[-1]
			del row[-1]
			del row[-1]
			del row[-1]
			del row[-3]
			del row[-5]
			#print(row)
	else:
		for row in data:
			del row[0]

	# delete NA rows
	outputData = []
	outputLabel = []
	print(len(data))
	for i in range(0, len(data)):
		row = data[i]
		if 'NA' in row:
			#print('hit')
			#data.remove(row)
			#trainLabel.remove(trainLabel[i])
			pass
		else:
			outputData.append([float(i) for i in row])
			outputLabel.append(trainLabel[i])
		i+=1

	print('after deletion',len(outputData))


	return (outputData, outputLabel)

def main():
	# read in train dataset
	outputData, trainLabel = readData('haha.csv','train')
	numData = len(outputData)
	print('total # of rows',numData)	
	numLabels = len(trainLabel[0])
	print('# rows of labels', len(trainLabel))
	print('total # of labels:',numLabels)

	# casting
	trainData = tf.to_float(outputData)#list to tensor
	trainLabel = tf.to_float(trainLabel)

	# Reference to tensorflow doc&tutorial
	# Create the model
	x = tf.placeholder(tf.float32, [None, 17])
	W = tf.Variable(tf.zeros([17, numLabels]))
	b = tf.Variable(tf.zeros([numLabels]))
	y = tf.matmul(x, W) + b

	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, numLabels])

	cross_entropy = tf.reduce_mean(
	  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# Train
	for _ in range(100): # test for 100 epochs
		start = random.randint(0, numData-100) # pick a random consecutive batch of 100
		#print('start',startIndex)
		batch_xs = trainData[start:start+100].eval(session=sess)
		batch_ys = trainLabel[start:start+100].eval(session=sess)
		#batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# read in test file	
	testData, testLabel = readData('test.csv','test')
	numTestData = len(testData)
	#print(testData[0])
	print('total # of test rows',numTestData)
	# casting
	testData = tf.to_float(testData).eval(session=sess)
	testLabel = tf.to_float(testLabel).eval(session=sess)

	# run
	print(sess.run(accuracy, feed_dict={x: testData, y_: testLabel}))


	
main()
