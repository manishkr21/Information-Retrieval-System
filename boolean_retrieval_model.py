import pickle
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')    # define the stemmer of english
from nltk.corpus import stopwords
Stopwords = set(stopwords.words('english'))           # will give the stopwords of the english language


pg_list = pickle.load(open('pg_list.dat', 'rb'))
files_with_index = pickle.load(open('files_with_index.dat', 'rb'))
posting_list = pickle.load(open('posting_list.dat', 'rb'))


bool_query = input("search using boolean retrieval model ==>")

en_string = bool_query.encode("ascii", "ignore")    #will remove the non-ascii character and ignore the error
bool_query_up = en_string.decode().split()            # decode the string


bool_cap = [word for word in bool_query_up if word.isupper()]           # store only upper words

#take only those word which are not stopword,having length greator than one, not uppercase and do stemming there after
bool_low = [stemmer.stem(word) for word in bool_query_up if (len(word) > 1) and (word.lower() not in Stopwords) and not word.isupper()]

fnl_bool_query = bool_cap + bool_low                            # combine capital and small words

bool_out = {}
for word in fnl_bool_query:                                     
    for d,f in posting_list[word].items():
        if d not in bool_out.keys():
            bool_out[d] = f                                      # just take unique elements inside words
        else:
            bool_out[d] += f                                     # if repetition of word occur just add its frequency to same previous document

bool_fnl = sorted(bool_out.items(), key = lambda x : x[1], reverse = True)   # do sort the bool_out reversly accrd to values of bool_out

for i in bool_fnl[0:10]:
    print(files_with_index[i[0]])                               # print query relevant document name



