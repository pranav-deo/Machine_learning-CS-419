#class implementation of the previous code

import numpy as np
import random
import time
threshold_height = 5
#I'm assuming I will be getting two numpy arrays of size n*(m+1) array names are data and index
#also I have cross checked all the functions :)
class d_tree(object):
	"""docstring for d_tree"""
	def __init__(self, data,height):
		super(d_tree, self).__init__()
		#attributes of this class
		self.threshold_height = threshold_height
		self.isleaf = False
		self.left = None
		self.right = None
		self.data = data
		self.height = height
		self.pred = self.prediction(self.data)
		self.attribute_value = None
		self.corr = None
		self.split_attribute = None
		self.split_index = None

		self.left_data = None
		self.right_data = None
		self.standard_dev = (self.sd(self.data))[-1]
		

		# self.threshold_loss = None
		# print('height.....===',self.height)	
		# print('xxxxxxxxxxxx......Decision_Tree_called........xxxxxxxxxxx')

		#adding the condition for the split
		if(self.data.shape[0]==1 or self.standard_dev==0 or self.height== self.threshold_height ):
			self.isleaf= True
			# print('this is a leaf at height ===',self.height)
			# print('its prediction is ===',self.pred)

		if(self.data.shape[0]>1 and self.standard_dev>0 and self.isleaf==False):

			#print(self.sd(self.data))
			#print(self.standard_dev)
			self.corr = self.correlation(self.data)			
			self.split_attribute = self.find_attribute(self.corr)
			# print('split_attribute...===',self.split_attribute)
			#sort the data
			self.data = self.index_sort(self.data,self.split_attribute)
			# print('sorted_data.....')
			# print(self.data)

			self.split_index = self.find_split_index(self.data,self.split_attribute)
			#print('split_index...===',self.split_index)
			
			if(self.split_index==0):
				self.isleaf = True
				# print('this is a leaf at height ===',self.height)
				#print('its prediction is ===',self.pred)

		if(self.height<=self.threshold_height):
			if(self.isleaf==False):
				# print(' ')
				self.attribute_value = self.data[self.split_index,self.split_attribute]
				# print('......attribute_value......===',self.attribute_value)
				self.left_data,self.right_data = self.split(self.data,self.split_index)
				# print('left child_data.....')
				# print(self.left_data)
				# print('right child_data.....')
				# print(self.right_data)
				self.left = d_tree(self.left_data,(self.height+1))
				# print(' ')
				self.right = d_tree(self.right_data,(self.height+1))
				
	#member functions
	def mean(self,a):
		# finds mean of each column
		mean = np.sum(a,axis=0)
		mean = mean/(a.shape)[0]
		return mean

	def mean_dev(self,a):
		#mean_dev = x-x'
		avg = self.mean(a)
		b = a-avg
		return b

	def sd(self,a):
		#no 1/n-1 term in sd
		b = self.mean_dev(a)
		b = np.multiply(b,b)
		b = np.sum(b,axis=0)
		b = np.sqrt(b)
		return b

	def correlation(self,a):
		#returns an 1*(m+1) array which has the correlations of each attribute w.r.t the last column
		#the last element of this array would be 1
		dev_mean = self.mean_dev(a)
		s_dev = self.sd(a)
		corr = dev_mean.T*dev_mean.T[-1]
		corr = corr.T
		temp = s_dev*s_dev[-1]
		corr = corr/temp
		corr = np.sum(corr, axis=0)
		for i in range(0,corr.shape[0]):
			if(corr[i]!=corr[i]):
				corr[i]=0
				#print('............NAN_OCCURED.............')
		# if(corr!=corr):
		
		#print(corr)
		return corr

	def find_attribute(self,a):
		#returns the index no. the attribute index. input must be the correlation matrix
		a = a[:-1]
		a = np.absolute(a)
		b = np.argmax(a)
		return b

	def add_rid_sort(self,a,n):
		#this function adds the row id to the input array and then sorts it according to the attribute found
		#n is the id w.r.t which sorting needs to be done
		#input is the data_array and attribute_index
		l = (a.shape)[0]
		rid = np.arange(l).reshape(l, 1)
		b = np.concatenate((a,rid),axis=1)
		c = b[b[:,n].argsort()]
		return c

	def index_sort(self,a,n):
		#the above function without row_id sort
		#this function doesn't add row id to the input array but sorts it according to the attribute found
		#n is the id w.r.t which sorting needs to be done
		#input is the data_array and attribute_index
		c = a[a[:,n].argsort()]
		return c

	def prediction(self,a):
		#inputs the data array
		#prediction would be the mean of the last column 
		l = (a.shape)[0]
		b = a[:,-1]
		b = b.reshape(l,1)
		pred = self.mean(b)
		return pred[0]

	def loss(self,a):
		#for this function given an numpy array it calculates the pred and returns the error
		#calculates the mean square loss
		l = (a.shape)[0]
		b = a[:,-1]
		b = b.reshape(l,1)
		loss = self.mean_dev(b)
		loss = np.multiply(loss,loss)
		loss = np.sum(loss, axis = 0)
		return loss

	def find_split_index(self,a,n):
		#got an array in which index n's attribute is sorted
		#now need to produce the best split
		total_loss = []
		l = a.shape[0]
		for i in range(0,l-1):
			loss = 0.0
			#need to add end conditions
			if(i==0):
				loss = loss + self.loss(a)
			else:
				loss = loss + self.loss(a[0:i]) + self.loss(a[i:l])
			total_loss.append(loss)
		total_loss = np.asarray(total_loss)
		split_index = np.argmin(total_loss)
		return split_index
	
	def split(self,a,m):
		#calculate current loss
		#we should have the split index and sorted array(important) and attribute index
		l = a.shape[0]
		return a[0:m],a[m:l]

def insert_pred(item, tree):
    if(tree.isleaf==True):
    	prediction = tree.pred
    	return prediction

    else:
    	if(item[tree.split_attribute] < tree.attribute_value):
    		return insert_pred(item,tree.left)
    	else:
    		return insert_pred(item,tree.right)

def calculate_loss(data,tree):
	l = data.shape[0]
	prediction =[]
	for i in range(0,l):
		pred = insert_pred(data[i],parent)
		prediction.append(pred)
	prediction = np.asarray(prediction)
	prediction = np.reshape(prediction,(l,1))
	#print(prediction)
	gt = data[:,-1]
	gt = np.reshape(gt,(l,1))
	
	loss = prediction - gt
	error = np.multiply(loss,loss)
	error = np.sum(error,axis=0)
	error = error[0]/(l)
	return error

def calculate_abs_loss(data,tree):
	l = data.shape[0]
	prediction =[]
	for i in range(0,l):
		pred = insert_pred(data[i],parent)
		prediction.append(pred)
	prediction = np.asarray(prediction)
	prediction = np.reshape(prediction,(l,1))
	#print(prediction)
	gt = data[:,-1]
	gt = np.reshape(gt,(l,1))
	
	loss = prediction - gt
	error = np.absolute(loss)
	error = np.sum(error,axis=0)
	error = error[0]/(l)
	return error


if __name__=='__main__':
	#instantize the class from the csv file
	#we get the data from paramters or infer.py
	start = time.time()
	data_array = np.genfromtxt('kaggle2_train.csv',delimiter=',')
	# print(data_array)
	l = data_array.shape[0]
	data_array = data_array[1:l]
	
	pruning_const = 20
	#split the  data array into train and val
	#generate a random number for dataset split into val and train
	#n-fold cross validation where n is pruning_const
	
	val_array_list = []
	train_array_list = []


	random_row_number = random.randrange(1,(data_array.shape[0]*2)/3,1)
	# print(random_row_number)

	#Making train and val datasets of sizes 2/3 and 1/3 of orignal datasets respectively
	train_array = data_array[0:random_row_number,:]
	val_array = data_array[random_row_number:int(random_row_number+data_array.shape[0]*1/3),:]
	train_array = np.vstack((train_array,data_array[int(random_row_number+data_array.shape[0]*1/3)+1:data_array.shape[0],:]))

	#pruning part
	train_loss_list = []
	val_loss_list = []
	
	for i in range(1,pruning_const):
		global threshold_height
		threshold_height = i
		#fit the tree
		print('height == ',i)
		parent = d_tree(train_array,0)
		train_loss = calculate_loss(train_array,parent)
		val_loss = calculate_loss(val_array,parent)
		# train_loss = calculate_loss(train_array_list[i-1],parent)
		# val_loss = calculate_loss(val_array_list[i-1],parent)
		print('train_error for height ',i,' === ',train_loss)
		print('val_error for height ',i,' === ',val_loss)
		train_loss_list.append(train_loss)
		val_loss_list.append(val_loss)
	
	best_height = np.argmin(val_loss_list)+1
	
	#since we got the best height from pruning now train it back again on all the data
	global threshold_height
	threshold_height = best_height
	#fit the tree
	parent = d_tree(data_array,0)
	print(' ')
	print('height of the final_tree is ','=== ',best_height)
	print('training error on the final_tree in ','=== ',calculate_loss(data_array,parent))
	end = time.time()
	
	print(end-start)
