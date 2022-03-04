#!/usr/bin/env python3
import pickle
import math
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')    # define the stemmer of english
from nltk.corpus import stopwords
Stopwords = set(stopwords.words('english'))           # will give the stopwords of the english language

from collections import defaultdict

#load required files
pg_list = pickle.load(open('pg_list.dat', 'rb'))       
files_with_index = pickle.load(open('files_with_index.dat', 'rb'))


try:
    # trying to load the required files
    posting_list = pickle.load(open('posting_list.dat', 'rb'))    
    doc_list = pickle.load(open('doc_list.dat', 'rb'))
    doc_idf = pickle.load(open('doc_idf.dat', 'rb'))
except:
    posting_list = defaultdict(defaultdict(int).copy)     #declare dictionary of dictionary

    for d in pg_list:                                    # if file is not there the create one
        for word,freq in pg_list[d].items():
            posting_list[word][d] = freq 


    no_of_doc = len(files_with_index)                    # find number of document
    doc_idf = defaultdict(defaultdict(int).copy)
    doc_list = defaultdict(defaultdict(int).copy)       # declare doc_list of type dict of dict
    for w in posting_list:
        for d,f in posting_list[w].items():
            doc_idf[d][w] = math.log2(no_of_doc/(len(posting_list[w])+1))
            doc_list[d][w] = f * math.log2(no_of_doc/(len(posting_list[w])+1))   # find the tf_idf of each term of particular document 


    #storing the files for further use
    pickle.dump(posting_list,open('posting_list.dat','wb'))
    pickle.dump(doc_idf, open('doc_idf.dat','wb'))
    pickle.dump(doc_list, open('doc_list.dat','wb'))




norm = {}
for d in doc_list:   
    s = 0
    for t in doc_list[d]:
        s += doc_list[d][t] * doc_list[d][t]      # sum the tf-idf of all terms in a document              
    norm[d] = math.sqrt(s)                        # find norm of each document



def tf_idf(query_for_doc):
   
    encoded_string = query_for_doc.encode("ascii", "ignore")    #will remove the non-ascii character and ignore the error
    query_doc = encoded_string.decode().split()            # decode the string
    
    query_cap = [word for word in query_doc if word.isupper() ]     #take only uppercase words
    
    #take only those word which are not stopword,having length greator than one, not uppercase and do stemming there after
    query_lower = [stemmer.stem(word) for word in query_doc if (len(word) > 1) and (word.lower() not in Stopwords) and not word.isupper()]
    
    query_to_doc = query_cap + query_lower                           #combine uppercase and lowercase word
    
    
    p_list_size = []                    # document frequency - no of documents in corpus, a term is appear
    idf = []                            # idf(d) = sum of log base 2 (no_of_doc/tf(t))
    score = []                          # make a list
    no_of_doc = len(files_with_index)
    
    for i in range(len(query_to_doc)):
        p_list_size.append(len(posting_list[query_to_doc[i]]))      # find term frequency 
        idf.append(math.log(2,(no_of_doc/(p_list_size[i]+1))))      # find idf of each term
        score += [(pg_no, tf * idf[i]) for pg_no,tf in posting_list[query_to_doc[i]].items()]   # find tf-idf of each document
    
    temp = {}
    for it in score:
        if it[0] not in temp.keys():     # take only those document's tf-idf which are which are not in temp dict
            temp[it[0]] = it[1]
        else:
            temp[it[0]] += it[1]         # repeated tf-idf got added to previous tf-idf of same doc
           
    
    
    cosine_sim = {doc_id: (tf_idf ) / norm[doc_id] for doc_id, tf_idf in temp.items()}      # find cosin similarity 
    cosine_sim = sorted(cosine_sim.items(), key = lambda x : x[1] ,reverse = True)          # sort cosine similarity accr to their normalized tf-idf score
    # return cosine_sim[0:10]
    

    ans = []
    for i in cosine_sim[0:5]:
        ans.append(files_with_index[i[0]])         # print relevant documents 
    return ans
