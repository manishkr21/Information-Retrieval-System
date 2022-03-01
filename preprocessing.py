import glob
import os             # to do the operating system related tasks
import numpy as np    # to do complex mathematical operations
import pandas as pd   # to do data manipulation related tasks
import math
import pickle
from tqdm.notebook import tqdm   # to check the running time
from collections import defaultdict


# nltk is a library which provides packages for lemmatizing and tokenizing words
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')    # define the stemmer of english

Stopwords = set(stopwords.words('english'))           # will give the stopwords of the english language



idx = 1
files_with_index = {}     # to store the index of each file
# dict_global = []         # to store the unique words
capwrd = []              # to store the seperate english word

pg_list = defaultdict(defaultdict(int).copy)

folpath = os.getcwd()+"\english-corpora"     # base path of the given english corpora
for file in tqdm(glob.glob("{0}\*".format(folpath))):   # to traverse the whole corpora

    fname = file
    file = open(file , "r",  encoding="utf8")          # open the file in read mode
    text = file.read()                                 # read the file
    
    encoded_string = text.encode("ascii", "ignore")    #will remove the non-ascii character and ignore the error
    decode_string = encoded_string.decode()            # decode the string
    
#     sentences = sent_tokenize(decode_string)       # convert paragrapgh into sentence
    words = word_tokenize(decode_string)           # convert sentence into word
    
    capwrd = [word for word in words if word.isupper() and any([l.isalnum() for l in word])]       #seperate the capital words
    
    #take only those word which are not stopword,having length greator than one, not uppercase and do stemming there after
    words = [stemmer.stem(word) for word in words if (len(words) > 1) and (word.lower() not in Stopwords) and not word.isupper() and any([l.isalnum() for l in word])]
    
    words.extend(capwrd)                         # combine the capital and lower words
    #words_unique = set(words)                                        # just take unique elements inside words
   # word_freq = {word: words.count(word) for word in words_unique }   # make a dictionary having word as key, and word frequency as a value
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1
    
    pg_list[idx] = word_freq                      # make a dictionary of dictionary having useful info

    files_with_index[idx] = os.path.basename(fname)    # make a dictionary having key as a doc number and value as a doc id
    idx = idx + 1



posting_list = defaultdict(defaultdict(int).copy)     #declare dictionary of dictionary

for d in tqdm(pg_list):
    for word,freq in pg_list[d].items():
        posting_list[word][d] = freq 



no_of_doc = len(files_with_index)                    # find number of document
doc_idf = defaultdict(defaultdict(int).copy)
doc_list = defaultdict(defaultdict(int).copy)    # declare doc_list of type dict of dict
for w in tqdm(posting_list):
    for d,f in posting_list[w].items():
        doc_idf[d][w] = math.log2(no_of_doc/(len(posting_list[w])+1))
        doc_list[d][w] = f * math.log2(no_of_doc/(len(posting_list[w])+1))   # find the tf_idf of each term of particular document 




pickle.dump(pg_list, open('pg_list.dat', 'wb'))
pickle.dump(files_with_index, open('files_with_index.dat', 'wb'))
pickle.dump(posting_list, open('posting_list.dat','wb'))
pickle.dump(doc_idf, open('doc_idf.dat', 'wb'))
pickle.dump(doc_list, open('doc_list.dat','wb'))