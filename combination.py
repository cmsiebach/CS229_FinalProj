import os, sys
from pysam import FastaFile
import pandas as pd
import numpy as np



def main():
	min_region_size = 1000
   	cell_types = []
    	cell_types.append("A549")
    	cell_types.append("GM12878")
    	cell_types.append("K562")

	train_matrix1 = "/srv/scratch/zho/" + str(min_region_size) + "_" + cell_types[0] + "_train_matrix.csv"
        output_vector1 = "/srv/scratch/zho/" + str(min_region_size) + "_" + cell_types[0]  + "_output_vector.csv"
        X1 = pd.DataFrame.from_csv(train_matrix1)
        Y1 = pd.Series.from_csv(output_vector1)
	print(X1)
	print(Y1)

	train_matrix2 = "/srv/scratch/zho/" + str(min_region_size) + "_" + cell_types[1] + "_train_matrix.csv"
        output_vector2 = "/srv/scratch/zho/" + str(min_region_size) + "_" + cell_types[1]  + "_output_vector.csv"
        X2 = pd.DataFrame.from_csv(train_matrix2)
        Y2 = pd.Series.from_csv(output_vector2)

	train_matrix3 = "/srv/scratch/zho/" + str(min_region_size) + "_" + cell_types[2] + "_train_matrix.csv"
        output_vector3 = "/srv/scratch/zho/" + str(min_region_size) + "_" + cell_types[2]  + "_output_vector.csv"
        X3 = pd.DataFrame.from_csv(train_matrix3)
        Y3 = pd.Series.from_csv(output_vector3)

	X = pd.DataFrame()
	Y = pd.Series()
	print(X)
	X = X.append(X1)
	print(X)
	X = X.append(X2)
	X = X.append(X3)
	Y = Y.append(Y1)
	Y = Y.append(Y2)
	Y = Y.append(Y3)
	Y.index = range(len(Y))
	X.index = range(len(X))
	# create some X
	X.to_csv("/srv/scratch/zho/" + str(min_region_size) + "_combination_train_matrix.csv");
	Y.to_csv("/srv/scratch/zho/" + str(min_region_size) + "_combination_output_vector.csv");

	return

if __name__ == '__main__':
	main()




