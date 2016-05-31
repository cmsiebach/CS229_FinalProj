import os, sys
import numpy as np
import pandas as pd
import error_metrics as em
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
import sklearn.metrics as metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mile import get_attributes_map
from sklearn.externals import joblib

def get_balanced_data(X):
    #sort into a promoter matrix and an enhancer matrix
    X_p = X[X['Labels'].isin([1])]
    X_p.index = range(len(X_p))
    X_e = X[X['Labels'].isin([0])]
    X_e.index = range(len(X_e))
    
    #take a random sample of the larger promoter matrix
    #such that it is the same length as the enhancers
    
    X_p_sample = X_p.sample(len(X_e))
    #X_p_short = X_p.iloc[0:len(X_e),:]
    #combine the promoters and enhancers back into one matrix	
    X_balanced = X_p_sample.append(X_e)
   # X_balanced = X_p_short.append(X_e)
    X_balanced.index = range(len(X_balanced))

    #Shuffle the data, such that it is not all promoters first
    X_balanced_shuffled = X_balanced.reindex(np.random.permutation(X_balanced.index))
    X_balanced_shuffled.index = range(len(X_balanced_shuffled))

    return X_balanced_shuffled

def trainModel(X_train, Y_train, X_test, Y_test, model="NB"):
    if model == "NB":
	clf = GaussianNB()
    elif model == "RF":
	clf = ensemble.RandomForestClassifier()
    elif model == "GB":
	clf = ensemble.GradientBoostingClassifier(learning_rate=0.1,
	                                          n_estimators=100,verbose=1)
    clf.fit(X_train, Y_train)
    Y_score = clf.predict_proba(X_test)
    auc = em.get_roc(Y_test,Y_score[:,1])
    return clf, auc

def print_important_params(model_type, clf, k):
    if model_type == "NB":
	imp_features = pd.Series(clf.theta_[1,:])
    else:
	imp_features = pd.Series(clf.feature_importances_)

    attributes_map = get_attributes_map(['A','C','G','T'],k)
    reverse_attributes_map = pd.Series(attributes_map.index)
    
    top_ten_weights = imp_features.sort_values(ascending=False).head(10)
    top_ten_kmers = reverse_attributes_map.iloc[top_ten_weights.index]
    print("Top ten important features:")
    print(top_ten_weights)
    print("Top 10 kmers:")
    print(top_ten_kmers)

def main():

    cell_types = []
    cell_types.append("A549")
    cell_types.append("GM12878")
    cell_types.append("K562")
    cell_types.append("combination")

    for i in range(0, 1):
	k=6
	trained_cell_type = cell_types[3] # training on combination/all data (A549, GM12878, K562)
	min_region_size = 1000
	train_matrix = "/srv/scratch/zho/" + str(min_region_size) + "_" + trained_cell_type + "_train_matrix.csv"
	output_vector = "/srv/scratch/zho/" + str(min_region_size) + "_" + trained_cell_type  + "_output_vector.csv"
        X = pd.DataFrame.from_csv(train_matrix)
	X['Labels']= pd.Series.from_csv(output_vector)
    	X_balanced_shuffled = get_balanced_data(X)
	
	kf = KFold(len(X_balanced_shuffled),n_folds=6)
	cur_auc = 0
	for train, test in kf:
	    df = X_balanced_shuffled
	    m,n = df.shape
	    new_clf, new_auc = trainModel(df.iloc[train,0:n-1], df.iloc[train,n-1],
					  df.iloc[test,0:n-1], df.iloc[test,n-1],sys.argv[1])
	    if new_auc > cur_auc:
		cur_auc = new_auc
		best_clf = new_clf
	  	print("updating best auc to:")
		print(cur_auc)
	
	#save model to file
	joblib.dump(best_clf, "/srv/scratch/zho/models/" + str(sys.argv[1]) + "_model.pkl")
	print_important_params(sys.argv[1], best_clf, k)
    	
    return

if __name__ == '__main__':
    main()
