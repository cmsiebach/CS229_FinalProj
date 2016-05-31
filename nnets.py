import os, sys

from keras.models import Sequential 
from keras.layers.core import Dense, Flatten, Activation 
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

from pysam import FastaFile
import error_metrics as em

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold

from pyDNAbinding.sequence import one_hot_encode_sequences
from pyTFbindtools.cross_validation import ClassificationResult

MAX_NUM_PEAKS = 500000

def iter_peaks_and_labels(fname):
    with open(fname) as fp:
        for i, line in enumerate(fp):
            if i >= MAX_NUM_PEAKS: break
            data = line.split()
            if len(data) < 3: continue
            yield (data[0], int(data[1]), int(data[2])), data[3]
    return

def load_sequences_and_labels(regions_fname, genome_fa_fname, balanced):
    seqs, labels = [], []
    min_region_size = 1000
    genome = FastaFile(genome_fa_fname)
    for region, label in iter_peaks_and_labels(sys.argv[1]):
        # create a new region exactly min_region_size basepairs long centered on 
        # region  
        expanded_start = region[1] + (region[2] - region[1])/2 - min_region_size/2
        expanded_stop = expanded_start + min_region_size
        region = (region[0], expanded_start, expanded_stop)
        seqs.append(genome.fetch(*region))

        if label == 'promoter': labels.append(1)
        elif label == 'enhancer': labels.append(0)
        else: assert False

    # crew code begin to balance data
    if balanced:
    	sequences = pd.DataFrame(seqs)
    	sequences['Labels'] = pd.Series(labels)
    	p_seqs = sequences[sequences['Labels'].isin([1])]
    	p_seqs.index = range(len(p_seqs))
    	e_seqs = sequences[sequences['Labels'].isin([0])]
    	e_seqs.index = range(len(e_seqs))

    	p_seqs_sample = p_seqs.sample(len(e_seqs))
   
    	balanced_seqs = p_seqs_sample.append(e_seqs)
    	balanced_seqs.index = range(len(balanced_seqs))
	
	shuffled_balanced_seqs = balanced_seqs.reindex(np.random.permutation(balanced_seqs.index))
	shuffled_balanced_seqs.index = range(len(shuffled_balanced_seqs))

    	return one_hot_encode_sequences(shuffled_balanced_seqs.iloc[:,0].as_matrix())[:,None,:,:],np.array(shuffled_balanced_seqs['Labels'].as_matrix())
    return one_hot_encode_sequences(seqs)[:,None,:,:],np.array(labels)


def build_model():
    # model definition
    model = Sequential()
    model.add(
        Convolution2D(100, 32, 4, border_mode='valid', input_shape=(1, 1000, 4))
    )
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=1))
    model.add(Activation("sigmoid"))

    # compile the model, and tell it what optimizer to use
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam())
    return model

def test_model(model, test_seqs, test_labels):

    proba = model.predict_proba(test_seqs, batch_size=50)
    pred_labels = model.predict_classes(test_seqs, batch_size=50)
    auc = em.get_roc(test_labels, proba)
    return auc

def main():
    seqs, labels = load_sequences_and_labels(sys.argv[1], sys.argv[2], True)
    print seqs.shape
    print labels.shape
    
    test_seqs, test_labels = load_sequences_and_labels(sys.argv[1], sys.argv[2], True)
    all_seqs, all_labels = load_sequences_and_labels(sys.argv[1], sys.argv[2], False)
    
    kf = KFold(len(seqs), n_folds=10)
    cur_auc = 0
    for train, test in kf:
    	model = build_model()
    	model.fit(seqs[train], labels[train], nb_epoch=30, batch_size=50)
        new_auc = test_model(model, seqs[test], labels[test])
 
	if new_auc > cur_auc:
	    cur_auc = new_auc
	    best_model = model
	    print(cur_auc)

    proba = best_model.predict_proba(seqs, batch_size=50)
    pred_labels = best_model.predict_classes(seqs, batch_size=50)

    
    test_proba = best_model.predict_proba(test_seqs, batch_size=50)
    test_pred_labels = best_model.predict_classes(test_seqs, batch_size=50)
    
    full_proba = best_model.predict_proba(all_seqs, batch_size=50) 
    full_pred_labels = best_model.predict_classes(all_seqs, batch_size=50)
    
    print("trained and tested on same, balanced data")
    print ClassificationResult(labels, pred_labels, proba)
    
    print("trained on balanced data, tested on another set of balanced data")
    print ClassificationResult(test_labels, test_pred_labels, test_proba)
    
    print("trained on balanced data, tested on all data available") 
    print ClassificationResult(all_labels, full_pred_labels, full_proba)
    
    json_string = best_model.to_json()
    open('ALL_model_architecture.json', 'w').write(json_string)
    best_model.save_weights('ALL_model_weights.h5')

if __name__ == '__main__':
    main()

# TODO - use numpy.save to cache the one hot encoded sequences
# 
# sample command:
# THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32 \
#   python2 CS229_FinalProj/test_nnet.py \
#       ./RAMPAGE_peaks/train_data/GM12878.labeled.bed \
#        ./RAMPAGE_peaks/train_data/GRCh38.genome.fa
#
#
