import os, sys
from pysam import FastaFile
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

# note: data[0] = chr, data[1] = start, data[2] = stop, data[3] = classification
def iter_peaks_and_labels(fname):
    with open(fname) as fp:
        for line in fp:
            data = line.split()
	    yield (data[0], int(data[1]), int(data[2])), data[3]
    return

def listKmers(string, bases, vec, k):
    if len(string) == k:
	vec.append(string)
    else:
	for b in bases:
	    listKmers(string + b, bases, vec, k)
    

def createAttributeMatrix(data, k, attributes_map):
    #train_file = open("trainMatrix.txt"
    length = (4**k)
    X = np.zeros((len(data), length))
    for i in range(0, len(data)):
	print("Starting sequence number:")
	print(i)
	sequence = data.loc[i]
	for j in range(0, len(sequence) - k):
	    kmer = sequence[j : j + k]
	    attribute = attributes_map.loc[kmer]
	    X[i, attribute] = X[i, attribute] + 1	    
    return pd.DataFrame(X);

def get_attributes_map(bases,k):
    kmers = []
    listKmers("", bases ,kmers, k)
    return pd.Series(range(0,4**k), index = kmers)

def main():
    
    min_region_size = 1000
    cell_type = "combination"
    genome = FastaFile("/srv/scratch/zho/GRCh38.genome.fa")
    
    k = 8 #k in kmer -- we choose 6
    sequence_list = []
    labels_list = []
    attributes_map = get_attributes_map(['A','C','G','T'],k)

    for region, label in iter_peaks_and_labels(sys.argv[1]):
        # create a new region exactly min_region_size basepairs long centered on 
        # region  
        expanded_start = region[1] + (region[2] - region[1])/2 - min_region_size/2
        if expanded_start < 0:
	    expanded_start = 0
	expanded_stop = expanded_start + min_region_size
        region = (region[0], expanded_start, expanded_stop)
        print region, label
	# note: 1 = promoter, 0 = enhancer
        if label == "promoter":
	    labels_list.append(1)
	else:
	    labels_list.append(0)
        print genome.fetch(*region)
	sequence_list.append(genome.fetch(*region))

    sequence_series=pd.Series(sequence_list)
    X = createAttributeMatrix(sequence_series, k, attributes_map)

    X.to_csv("/srv/scratch/zho/" + str(min_region_size) + "_" + cell_type + "_" + str(k)  + "mer_train_matrix.csv");
    labels_series = pd.Series(labels_list)
    labels_series.to_csv("/srv/scratch/zho/" + str(min_region_size) + "_" + cell_type + "_" + str(k) + "mer_output_vector.csv");

    return

if __name__ == '__main__':
    main()
