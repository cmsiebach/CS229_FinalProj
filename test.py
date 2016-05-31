import os, sys
import numpy as np
import pandas as pd
import nnets as nn
import error_metrics as em
import classify as cls
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from keras.models import model_from_json
from sklearn.externals import joblib
from pysam import FastaFile
from pyDNAbinding.sequence import one_hot_encode_sequences
from pyTFbindtools.cross_validation import ClassificationResult

def get_test_data(test_cell_type, min_region_size):
    test_output_vector = "/srv/scratch/zho/" + str(min_region_size) + "_" + test_cell_type  + "_output_vector.csv"
    test_train_matrix = "/srv/scratch/zho/" + str(min_region_size) + "_" + test_cell_type + "_train_matrix.csv"
    test_X = pd.DataFrame.from_csv(test_train_matrix)
    test_Y = pd.Series.from_csv(test_output_vector)
    return test_X, test_Y

def test_on(clf, trained_cell_type, test_cell_type, min_region_size, test_X, test_Y):
    predictions = clf.predict(test_X)
    y_score = clf.predict_proba(test_X)
    fpr, tpr, thresholds = metrics.roc_curve(test_Y, y_score[:, 1])
    fig=plt.figure()
    plt.plot(fpr, tpr, label='test')
    #fig.savefig("test.jpg")
    'em.overall_accuracy(test_Y, predictions)'
    em.norm_confusion_matrix(test_Y, predictions)
    #fname = "/srv/scratch/zho/roc_curves/" + trained_cell_type + "_on_" + test_cell_type + "_" + str(min_region_size) + "RF.jpg"
    #title = trained_cell_type + " testing on " + test_cell_type + " with Min Region " + str(min_region_size)
    'em.print_roc(test_Y, predictions)'
    'em.plot_roc(test_Y, predictions, fname, title)'

def main():

    cell_types = []
    cell_types.append("A549")
    cell_types.append("GM12878")
    cell_types.append("K562")
    cell_types.append("combination")

    min_region_size = 1000

    #LOAD MODELS:
    #Neural Nets
    NN_model = model_from_json(open('ALL_model_architecture.json').read())
    NN_model.load_weights('ALL_model_weights.h5')
    #Naive Bayes
    NB_model = joblib.load('/srv/scratch/zho/models/NB_model.pkl')
    #Random Forest
    RF_model = joblib.load('/srv/scratch/zho/models/RF_model.pkl')
    #Gradient Boosting
    GB_model = joblib.load('/srv/scratch/zho/models/GB_model.pkl')


    seqs, labels = nn.load_sequences_and_labels(sys.argv[1], sys.argv[2], True)
    print seqs.shape
    print labels.shape

    test_seqs, test_labels = nn.load_sequences_and_labels(sys.argv[1], sys.argv[2], True)
    all_seqs, all_labels = nn.load_sequences_and_labels(sys.argv[1], sys.argv[2], False)

    # NN_model
    proba = NN_model.predict_proba(seqs, batch_size=50)
    pred_labels = NN_model.predict_classes(seqs, batch_size=50)

    test_proba = NN_model.predict_proba(test_seqs, batch_size=50)
    test_pred_labels = NN_model.predict_classes(test_seqs, batch_size=50)

    full_proba = NN_model.predict_proba(all_seqs, batch_size=50)
    full_pred_labels = NN_model.predict_classes(all_seqs, batch_size=50)

    print("FOR SAME TRAIN AND TEST")
    print ClassificationResult(labels, pred_labels, proba)

    print("FOR OTHER BALANCED SET")
    print ClassificationResult(test_labels, test_pred_labels, test_proba)

    print("FOR FULL TEST")
    print ClassificationResult(all_labels, full_pred_labels, full_proba)

    # get test data in a form recognizable by NB, RF, and GB models
    trained_cell_type = cell_types[3]
    test_cell_type = cell_types[3]

    test_X, test_Y = get_test_data(test_cell_type, min_region_size)

    print("Error analysis for NB_model:")
    test_on(NB_model, trained_cell_type, test_cell_type, min_region_size, test_X, test_Y)
    
    print("Error analysis for RF_model:")
    test_on(RF_model, trained_cell_type, test_cell_type, min_region_size, test_X, test_Y)
    
    print("Error analysis for GB_model:")
    test_on(GB_model, trained_cell_type, test_cell_type, min_region_size, test_X, test_Y)
 
    return

if __name__=='__main__':
    main()
