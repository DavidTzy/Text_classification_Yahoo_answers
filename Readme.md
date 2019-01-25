1.This repository is about text classification on yahoo answers including the use of different sentence representation(Word2Vec,Doc2Vec)
and different models(NN,CNN,mult_NB). Doc2Vec+NN performs best.

Further implementation:
(1) RNN based models(LSTM,GRU), namely Bi-GRU model with two level attention mechanism should perform better than Doc2Vec+NN according to the experiments in the paper "Hierarchical Attention Networks for Document Classification", the structure of it is novel but complicated, should be time-cosuming in training.

(2) DCNN, a CNN based model proposed by Oxford Univerity gives another aspect of imploying CNN in NLP cases with no restriction on input lengths.

Still under testing since the dataset is too huge(1.4 million).


