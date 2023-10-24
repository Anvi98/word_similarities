import nltk
import numpy as np
import spacy
import re
from modules import *
from nltk.corpus import stopwords
from more_itertools import locate
nlp = spacy.load('en_core_web_sm')


# Numpy data
print("Loading Numpy data ...")
unique_tokens = np.load("numpy_data/unique_tokens.npy")
w_in_context = np.load("numpy_data/w_in_context.npy")
raw_freq = np.load("numpy_data/raw_freq.npy")
rel_freq = np.load("numpy_data/rel_freq.npy")
tf = np.load("numpy_data/tf.npy")
idf_list = np.load("numpy_data/idf_list.npy")
tfidf_list = np.load("numpy_data/tfidf_list.npy")
print("All import Done...")