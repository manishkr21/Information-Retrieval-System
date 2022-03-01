import pickle
import math
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')    # define the stemmer of english
from nltk.corpus import stopwords
Stopwords = set(stopwords.words('english'))           # will give the stopwords of the english language


from tqdm.notebook import tqdm   # to check the running time
from collections import defaultdict


pg_list = pickle.load(open('pg_list.dat', 'rb'))
files_with_index = pickle.load(open('files_with_index.dat', 'rb'))
posting_list = pickle.load(open('posting_list.dat', 'rb'))
doc_list = pickle.load(open('doc_list.dat', 'rb'))
doc_idf = pickle.load(open('doc_idf.dat', 'rb'))


norm = {}
for d in tqdm(doc_list):   
    s = 0
    for t in doc_list[d]:
        s += doc_list[d][t] * doc_list[d][t]      # sum the tf-idf of all terms in a document              
    norm[d] = math.sqrt(s)                        # find norm of each document



l_avgg = 0                         # variable to store sum of lengths
l_d = {}
no_of_doc = len(files_with_index)                    # find number of document

for i in doc_list:
    l_d[i] = len(doc_list[i])      # length of each document
    l_avgg += len(doc_list[i])     # sum of length of each document
l_avg = l_avgg/no_of_doc           # average of length of each document


query_for_doc = input("search using BM25 model => ")     # search using bm25 model

encoded_string = query_for_doc.encode("ascii", "ignore")    #will remove the non-ascii character and ignore the error
query_doc = encoded_string.decode().split()            # decode the string

query_cap = [word for word in query_doc if word.isupper()]             # store only capital words

#take only those word which are not stopword,having length greator than one, not uppercase and do stemming there after
query_lower = [stemmer.stem(word) for word in query_doc if (len(word) > 1) and (word.lower() not in Stopwords) and not word.isupper()]

query_to_doc = query_cap + query_lower   # combine upper and lower words

# print(query_to_doc)
bm_25_score = []                               # list to store bm_25 score
k = 1.4
b = 0.75
for i in tqdm(query_to_doc):                   
    
    score = []
    
    for j in posting_list[i]:
            tf = pg_list[j][i]
            # apply the bm-25 formula and find the sum of that for this document
            score = [(doc_idf[j][i]) * ((k + 1) * tf)/(k * (1-b + b * l_d[j]/l_avg)  + tf) ]
            bm_25_score.append((j,sum(score)))       # append the sum of score into bm_25_score

temp = {}
for it in bm_25_score:
    if it[0] not in temp.keys():          # take only unique terms
        temp[it[0]] = it[1]
    else:
        temp[it[0]] += it[1]              # if term repeat add that into previous score of same doc

fnl_score = sorted(temp.items(), key = lambda x: (x[1],x[0]),reverse = True)    # sort the score accrd to their bm score

for i in fnl_score[0:10]:
    print(files_with_index[i[0]])         # print relevant documents 




