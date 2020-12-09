#imports
import numpy as np
import pandas as pd
import re, sys
from collections import Counter, defaultdict
from sklearn import feature_selection as fs
from sklearn import feature_extraction as fe
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def setUpData(train_df, test, c=3):
  train_size = len(train_df)
  Y = train_df[0]
  train = train_df.drop(columns=0)
  train = train.rename(columns={1:0})
  big = pd.concat([train, test])
  #print(big)
  docs = [cmer(n, c) for n in big[0]]
  train_mat = build_matrix(docs)
  csr_l2normalize(train_mat)
  test = train_mat[train_size:,]
  train = train_mat[0:train_size,]
  return train, test, Y

def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat

def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat
        
def namesToMatrix(names, c):
    docs = [cmer(n, c) for n in names]
    return build_matrix(docs)

def cmer(name, c=3):
    r""" Given a name and parameter c, return the vector of c-mers associated with the name
    """
    name = name.lower()
    if len(name) < c:
        return [name]
    v = []
    for i in range(len(name)-c+1):
        v.append(name[i:(i+c)])
    return v
def featureSelection2(X_train, Y_train, X_test):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X_train, Y_train)
    model = SelectFromModel(clf, prefit=True)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    return X_train, X_test

def featureExtraction(train_df, test_df):
  #feature extraction using bio library to acquire peptide attributes
  n = len(train_df)
  Y = train_df[0]
  train_df = train_df.drop(columns=0)
  train_df = train_df.rename(columns={1:0})
  big = pd.concat([train_df, test_df], ignore_index=True)
  big['molecular_weight'] = 0.0
  #big['flexibility'] = 0
  big['isoelectric_point'] = 0.0
  big['aromaticity'] = 0.0
  big['stability'] = 0.0
  for i in range (len(big)):
    #print(big.iloc[i, 0])
    val = big.iloc[i, 0]
    #invalid peptide check, set all values to 0
    if 'X' in val or 'Z' in val:
      big.at[i, 'molecular_weight'] = -1
      #big.at[i, 'flexibility'] = -1
      big.at[i, 'isoelectric_point'] = -1
      big.at[i, 'aromaticity'] = -1
      big.at[i, 'stability'] = -1
      continue
    model = ProteinAnalysis(val)
    big.at[i, 'molecular_weight'] = model.molecular_weight()
    #big.at[i, 'flexibility'] = model.flexibility()
    big.at[i, 'isoelectric_point'] = model.isoelectric_point() 
    big.at[i, 'aromaticity'] = model.aromaticity()
    big.at[i, 'stability'] = model.instability_index()
  big = big.drop(columns=0)
  train_df = big.iloc[:n,]
  test_df = big.iloc[n:,]
  return train_df, test_df, Y

def featureSelection(X_train, Y_train, X_test):
    #using selectKbest feature selection
  #bestFeatures = fs.
  bestFeatures = fs.SelectKBest(chi2, k=100)
  fit = bestFeatures.fit(X_train, Y_train)
  new_train = fit.transform(X_train)
  new_test = fit.transform(X_test)
  return new_train, new_test

def KNNClassifier(train_df, test_df, c=3, k=3):
  #X_train, X_test, Y_train = setUpData(train_df, test_df, c = c)
  X_train, X_test, Y_train = featureExtraction(train_df, test_df)
  X_train, X_test = featureSelection2(X_train, Y_train, X_test)

  model = KNeighborsClassifier(n_neighbors=k, metric='cosine')

  model.fit(X_train, Y_train)

  return model.predict(X_test)

def GaussianClassifier(train_df, test_df, c=3):
  X_train, X_test, Y_train = setUpData(train_df, test_df, c = c)
  X_train, X_test = featureSelection2(X_train, Y_train, X_test)
  model = GaussianNB()
  model.fit(X_train.todense(), Y_train)

  return model.predict(X_test.todense())

def ForestClassifier(train_df, test_df, c=3):
  #X_train, X_test, Y_train = setUpData(train_df, test_df, c = c)
  X_train, X_test, Y_train = featureExtraction(train_df, test_df)
  X_train, X_test = featureSelection2(X_train, Y_train, X_test)
  model = RandomForestClassifier()
  model.fit(X_train, Y_train)

  return model.predict(X_test)

#datasets to dataframes
train_df = pd.read_csv('train.dat', sep = '\t', header=None)
test_df = pd.read_csv('test.dat', sep = '\t', header=None)

#separating classifiers
Y = train_df[0]
X_dat = train_df[1]

#results = KNNClassifier(train_df, test_df, c=3, k=3)
results = ForestClassifier(train_df, test_df)
results_df = pd.DataFrame()
results_df['Results'] = results
results_df.to_csv('predictions.dat', index=False, header=None)