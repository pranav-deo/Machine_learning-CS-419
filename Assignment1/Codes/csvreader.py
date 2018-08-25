import csv
import numpy as np
import argparse

def readcsv(name):

	#attributes and rows list
	attributes = np.array([])
	rows = []

	#reading 
	with open(name,'r') as csvfile:
		csvreader = csv.reader(csvfile)
		attributes = csvreader.next()

		for row in csvreader:
			rows.append(row)
	rows = np.array(rows)

	print("attributes:")
	print(attributes)
	print("rows:")
	print(rows)

if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-n", "--name", required=True,help="path to the csvfile")
	args = vars(ap.parse_args())
	readcsv(args["name"])

