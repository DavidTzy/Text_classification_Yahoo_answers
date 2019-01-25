import os
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import keras
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.text import text_to_word_sequence,Tokenizer,hashing_trick
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras import callbacks
from gensim import corpora, models
from gensim.models import Doc2Vec, Word2Vec
from keras.utils import to_categorical
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from bs4 import BeautifulSoup
from sklearn import utils,svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from gensim.models import Doc2Vec


"""EDA Analysis"""
input_path='/Users/tanzhenyu/Desktop/stat841/data/'
data = pd.read_csv(os.path.join(input_path,'Text_classfication.csv'),header=None)

data.head()

print("The missing value for each column is\n{0} \nand the total missing value is {1}".format(data.isnull().sum(),data.isnull().sum().sum()))


#replacing NA value with empty string
data_raw=data.replace(np.nan, '', regex=True)

# check the shape 
print('The shape of data is {0}'.format(data_raw.shape))



#Join the question, title and answer together for each record 

data_raw['merged_text'] = data_raw.iloc[:,1]+data_raw.iloc[:,2]+data_raw.iloc[:,3]
data_raw.drop(data_raw.columns[[1,2,3]],axis = 1, inplace = True)

#Rename the col
data_raw.columns = ["tag",'reviews']


X_train_raw,X_test_raw,y_train,y_test=train_test_split(data_raw["reviews"],
                                               data_raw["tag"],
                                               test_size=0.2,random_state=32)


y_train_cat = to_categorical(np.asarray(y_train-1))


y_test_cat = to_categorical(np.asarray(y_test-1))





#Function to clean the text and only return the meaningful words
def review_to_words(raw_review,mode):
    
    review_text = BeautifulSoup(raw_review,"lxml").get_text()     
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    if mode == "words":
        return meaningful_words 
    elif mode == "sent":
        return " ".join(meaningful_words)

    




train_data = X_train_raw.apply(lambda x: review_to_words(x,mode ="sent"))
test_data = X_test_raw.apply(lambda x: review_to_words(x,mode ="sent"))





class KerasEmbeddingVectorizer(object):
    ''' 
    Keras offers us three method for sent representation: binary,count,tfidf   
    '''
    
    def __init__(self, vector_size ,mode):
        self.vector_size = vector_size
        self.mode =mode
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        tokenize =Tokenizer(num_words=self.vector_size)
        tokenize.fit_on_texts(X)
        X_train = tokenize.texts_to_matrix(X,mode = self.mode)
                
        return X_train



'''MeanWord2Vec'''




# could use the mode "words" if use review_to_words, here we just modifed a little from previous train_data 
train_reviews_w2v = train_data.apply(lambda r: r.split())
test_reviews_w2v = test_data.apply(lambda r: r.split())



class MeanEmbeddingVectorizer(object):
    def __init__(self,vector_size,X_dim):
        self.dim = vector_size
        self.X_dim = X_dim
            
    def fit(self, X, y):
        model = Word2Vec(X, size=self.dim)
        self.word2vec = dict(zip(model.wv.index2word,model.wv.vectors))
        return self 

    def transform(self, X):
        X = np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
        if self.X_dim == 2:
            return X
        elif self.X_dim == 3:
            return  np.expand_dims(X, axis=2)
            


'''TF.IDF Weighted Word2Vec'''


class TfidfEmbeddingVectorizer(object):
    def __init__(self, vector_size,X_dim):
        self.dim = vector_size
        self.X_dim = X_dim
        
    def fit(self, X, y):        
        model = Word2Vec(X, size=self.dim)
        self.word2vec = dict(zip(model.wv.index2word,model.wv.vectors))        
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        X = np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
        if self.X_dim == 2:
            return X
        elif self.X_dim ==3:
            return  np.expand_dims(X, axis=2)


'''Doc2Vec'''
#Apply the cleaning procedure


min_count = 5
negative = 5
dm = 0


train_raw = pd.DataFrame({"reviews":X_train_raw,"tag":y_train})
test_raw = pd.DataFrame({"reviews":X_test_raw,"tag":y_test})



class D2V(object):
    def __init__(self,vector_size,min_count,negative,dm,cores,dimension):
        
        self.dimension =dimension  
        self.min_count = min_count
        self.negative = negative
        self.dm = dm
        self.vector_size = vector_size
        self.cores= cores
        
    def fit(self, X, y):        
        X_tagged = X.apply(
                lambda r: TaggedDocument(words=review_to_words(r['reviews'],mode = "words"), tags=[r.tag]), axis=1)
        self.model = Doc2Vec(dm=self.dm, vector_size=self.vector_size, 
                             negative=self.negative, min_count=self.min_count,workers=self.cores,hs=0,
                             sample = 0)        
        self.model.build_vocab([x for x in tqdm(X_tagged.values)])
        for epoch in range(10):
            self.model.train(utils.shuffle([x for x in tqdm(X_tagged.values)]), total_examples=len(X_tagged.values), epochs=1)
            self.model.alpha -= 0.002
            self.model.min_alpha = self.model.alpha
        return self 

    def transform(self, X):
        sents = X.apply(
                lambda r: TaggedDocument(words=review_to_words(r['reviews'],mode = "words"), tags=[r.tag]), axis=1)
        regressors = [self.model.infer_vector(doc.words, steps=20) for doc in sents]    
        if self.dimension == 3:
           regressors = np.expand_dims(regressors, axis=2)
        elif self.dimension == 2:
           regressors = np.array(regressors)
        return regressors
    
    
def NN_model(vector_size):
    def model():
        model= Sequential([
                Dense(512,activation='relu',input_shape=(vector_size,)),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(128,activation='relu'),
                BatchNormalization(),
                
                Dense(10,activation='softmax')
                
                ]
                )
        model.compile(optimizer="Adam",
                     loss= "categorical_crossentropy",
                      metrics=['acc'])
    
        return model
    return model






def CNN_model(vector_size):
    def model():
        model= Sequential([
                Conv1D(128, 5, activation='relu',input_shape=(vector_size,1)),
                Conv1D(64, 5, activation='relu'),
                MaxPooling1D(10),
                
                Flatten(),
                Dense(64, activation='relu'),        
                Dense(10,activation='softmax')        
                ]
                )
        model.compile(optimizer='Adam',
                     loss='categorical_crossentropy',
                      metrics=['acc'])
        return model
    return model





#Fitting the models


#baseline model 


Model_baseline = Pipeline([("Keras", KerasEmbeddingVectorizer(vector_size=100,mode='binary')), 
                   ("MultiNB",MultinomialNB())])



parameters = {   
    'MultiNB__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  
    }  

Multi_NB = GridSearchCV(Model_baseline, parameters,
                   verbose=0, refit=True,cv=5)

Multi_NB.fit(train_reviews_w2v,y_train)
means = Multi_NB.cv_results_['mean_test_score'].mean()


score_baseline= [("baseline",means)]




batch_size =300

vector_size = 1000
cores = multiprocessing.cpu_count()

NN_meanW2V = Pipeline([("meanw2v",MeanEmbeddingVectorizer(vector_size=vector_size,X_dim=2)), 
                   ("NN",KerasClassifier(build_fn = NN_model(vector_size = vector_size),epochs=10, batch_size=batch_size,verbose=0
                           ))])
    


NN_TfidfWeightedW2V = Pipeline([("tfidf", TfidfEmbeddingVectorizer(vector_size=vector_size,X_dim=2)), 
                   ("NN",KerasClassifier(build_fn = NN_model(vector_size = vector_size),epochs=10, batch_size=batch_size,
                           verbose = 0))])



NN_Doc2Vec = Pipeline([("Doc2Vec", D2V(vector_size=vector_size,min_count=min_count,negative=negative,dm=1,cores=cores,dimension = 2)), 
                   ("NN",KerasClassifier(build_fn = NN_model(vector_size=vector_size),epochs=10, batch_size=batch_size,verbose=0))])



CNN_meanW2V = Pipeline([("meanw2v",MeanEmbeddingVectorizer(vector_size=vector_size,X_dim= 3)), 
                   ("CNN",KerasClassifier(build_fn = CNN_model(vector_size=vector_size),epochs=10, batch_size=batch_size,verbose=0
                          ))])


CNN_Doc2Vec = Pipeline([("Doc2Vec", D2V(vector_size=vector_size,min_count=min_count,negative=negative,dm=1,cores=cores,dimension = 3)), 
                   ("CNN",KerasClassifier(build_fn = CNN_model(vector_size=vector_size),epochs=10, batch_size=batch_size,verbose=0
                           ))])


CNN_TfidfWeightedW2V = Pipeline([("tfidf", TfidfEmbeddingVectorizer(vector_size=vector_size,X_dim=3)), 
                   ("CNN",KerasClassifier(build_fn = CNN_model(vector_size=vector_size),epochs=10, batch_size=batch_size,verbose=0
                         ))])


W2V_models = [
    ("meanw2v_NN", NN_meanW2V),
    ("tfidfweightedw2v_NN", NN_TfidfWeightedW2V),    
    ("tfidfweightedw2v_CNN", CNN_TfidfWeightedW2V),
    ("meanw2v_CNN", CNN_meanW2V),
]




D2V_models =[
        ("NN_Doc2Vec", NN_Doc2Vec),
        ("CNN_Doc2Vec", CNN_Doc2Vec),
        ]


from tabulate import tabulate

unsorted_scores_W2V = [(name, cross_val_score(model, train_reviews_w2v , y_train_cat, cv=5).mean()) for name, model in W2V_models]


unsorted_scores_D2V = [(name, cross_val_score(model, train_raw , y_train_cat, cv=5).mean()) for name, model in D2V_models]






scores = sorted(unsorted_scores_W2V + unsorted_scores_D2V+score_baseline, key=lambda x: -x[1])

print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))




NN_Doc2Vec.fit(train_raw , y_train_cat)

text_accuracy=NN_Doc2Vec.score(test_raw,y_test_cat)

