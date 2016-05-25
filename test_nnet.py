import os, sys

from keras.models import Sequential 
from keras.layers.core import Dense, Flatten, Activation 
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

from pysam import FastaFile

import numpy as np

from pyDNAbinding.sequence import one_hot_encode_sequences
from pyTFbindtools.cross_validation import ClassificationResult

MAX_NUM_PEAKS = 5000

def iter_peaks_and_labels(fname):
    with open(fname) as fp:
        for i, line in enumerate(fp):
            if i >= MAX_NUM_PEAKS: break
            data = line.split()
            if len(data) < 3: continue
            yield (data[0], int(data[1]), int(data[2])), data[3]
    return

def load_sequences_and_labels(regions_fname, genome_fa_fname):
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

    return one_hot_encode_sequences(seqs)[:,None,:,:], np.array(labels)


def build_model():
    # model definition
    model = Sequential()
    model.add(
        Convolution2D(100, 4, 32, border_mode='valid', input_shape=(1, 1000, 4))
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

def main():
    seqs, labels = load_sequences_and_labels(sys.argv[1], sys.argv[2])
    print seqs.shape
    print labels.shape
    model = build_model()
    model.fit(seqs, labels, nb_epoch=5, batch_size=50)
    proba = model.predict_proba(seqs, batch_size=50)
    pred_labels = model.predict_classes(seqs, batch_size=50)

    print ClassificationResult(labels, pred_labels, proba)

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
