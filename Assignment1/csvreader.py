#To run in terminal: python csvreader.py -n=path/to/dataset.csv
#gives : attributes array, array of attribute's data (with Row ID), array of mean of attribute's data, array of variance data (except output attribute)

#P.S. ------>This is same as infer.py

import csv
import numpy as np
import argparse
from decision_tree import *
import random
import pickle

class dtree:
 	
	def readcsv(self,name,arr1,arr2):

		#rows list
		columns= np.empty([])
		#attributes
		attributes = np.array([])

		rows = []

		#reading 
		with open(name,'r') as csvfile:
			csvreader = csv.reader(csvfile)
			attributes = csvreader.next()

			for row in csvreader:
				rows.append(row)
		rows = np.array(rows)
		columns = np.transpose(rows,None)
		
		np.resize(arr1,np.shape(columns))
		np.resize(arr2,np.shape(attributes))
		arr1=columns
		arr2=attributes
		arr1 = arr1.astype(np.float)
		return arr1,arr2


if __name__=='__main__':
	
	#Parsing the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-n", "--data_file", required=True,help="path to the csvfile")
	# parser.add_argument('--aboslute', const='absolute', 
	#for absolute or square error  #ap.add_argument("-t", "--data_file", required=False,help="path to the csvfile") #----------Change it to true afterwards!!!-------
	args = vars(ap.parse_args())

	#Data array
	cols=np.empty([])

	#Attributes array
	att=np.empty([])


	D=dtree()
	p=args["data_file"]

	#Filling arrays of data and attributes respectively
	cols,att = dtree.readcsv(D,p,cols,att)

#Loop around this for n-n cross validation
#______________________________________________________________________________________________________________

	#generating random number for dataset split into val and train
	
	random_row_number = random.randrange(1,cols.shape[1]*2/3,1)
	# print(random_row_number)

	#Making train and val datasets of sizes 2/3 and 1/3 of orignal datasets respectively
	train = cols[:,0:random_row_number]
	val = cols[:,random_row_number:int(random_row_number+cols.shape[1]*1/3)]
	train = np.hstack((train,cols[:,int(random_row_number+cols.shape[1]*1/3)+1:cols.shape[1]]))

	print("train",train)
	print("val",val)
	#Making tree on the basis of train dataset
	# parent = d_tree(train,0)

#_____________________________________________________________________________________________________________

	# #Saving the model
	# filename = 'model.csv'
	# pickle.dump(model, open(filename, 'wb'))
