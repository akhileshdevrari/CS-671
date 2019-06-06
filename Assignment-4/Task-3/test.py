# Reference: This code has been motivated from: https://medium.com/@david.campion/text-generation-using-bidirectional-lstm-and-doc2vec-models-1-3-8979eb65cb3a


from __future__ import print_function
#import Keras library
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy

#import spacy, and spacy french model
# spacy is used to work on text
import spacy
nlp = spacy.load('en')

#import other libraries
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
from six.moves import cPickle

#define input and output files
input_file = 'testText.txt'# data file containing raw text
save_dir = 'results/' # directory to store trained NN models
vocab_file = os.path.join(save_dir, "words_vocab.pkl")
sequences_step = 1 #step to create sequences
seq_length = 30 # sequence length



#load vocabulary
print("loading vocabulary...")
vocab_file = os.path.join(save_dir, "words_vocab.pkl")

with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
	words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)

from keras.models import load_model
# load the model
print("loading model...")
model = load_model(save_dir + "/" + 'my_model_generate_sentences.h5')


def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)



words_number = 50 # number of words to generate
seed_sentences = [] #seed sentences to start the generating.
seed_sentences.append("elizabeth wanted to run away in the distance because")
seed_sentences.append("mr. darcy was a very")
seed_sentences.append("mrs. bennet was so excited that she could")
seed_sentences.append("mr. wickham , a tall and young man , made")
seed_sentences.append("lady catherine , having heard about elizabeth and darcy , visits")


for seed_sentence in seed_sentences:
	#initiate sentences
	generated = ''
	sentence = []

	#we shate the seed accordingly to the neural netwrok needs:
	for i in range (seq_length):
		sentence.append("a")

	seed = seed_sentence.split()

	for i in range(len(seed)):
		sentence[seq_length-i-1]=seed[len(seed)-i-1]

	generated += ' '.join(sentence)

	#the, we generate the text
	for i in range(words_number):
		#create the vector
		x = np.zeros((1, seq_length, vocab_size))
		for t, word in enumerate(sentence):
		    x[0, t, vocab[word]] = 1.

		#calculate next word
		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, 0.33)
		next_word = vocabulary_inv[next_index]

		#add the next word to the text
		generated += " " + next_word
		# shift the sentence by one, and and the next word at its end
		sentence = sentence[1:] + [next_word]

	#print the whole text
	print('\n\n\n')
	print(generated)
