

Prerequisites:   
        install nltk(all files)		(a good site to follow  https://www.nltk.org/install.html)    
        note: just run command  ==> make req     (to dwonload and install the nltk and packages )

About the generating file:
        pg_list (included in module) : a dict of dict, in which data stored like       { doc_no :{ term1 : frequncy_of_term1, term2 : frequency_of_term2, ...}, ...}
        posting_list : a dict of dict, in which data stored like  { term1:{doc1:frequncy_of_term1, doc2:frequency_of_term2},... }
        doc_idf : a dict of dict, in which data stored like       {doc_no : { term1: idf_of_term1, term2 : idf_of_term2,..}, ...}
        doc_list : a dict of dict in which data stored like       {doc_no : { term1: tf_idf_of_term1, term2 : tf_idf_of_term2,..}, ...}
        files_with_index (included in module) : a dictionary in which key is document number(ex: 1,2,3,..) and its value is document id(ex: C00007.txt)

note: if pg_list, files_with_index files are not with module do run ===>   make preprocessing

About each file:
        preprocessing.py : do the required processing, remove the word in which all letters are non-ascii and generate two files pg_list and files_with_index
        boolean_retrieval_model.py : Actual file in which boolean_retrival model is implemented
        tf_idf_model.py : Actual file in which tf_idf model is implemented
        bm25_model.py : Actual file in which bm25_model is implemented
        run_model1.py : file to run the boolean_retrieval_model, will generate file name Boolean_Model_Output.txt having Qrels formats of five queries
        run_model2.py : file to run the tf_idf_model, will generate file name tfidf_Model_Output.txt having Qrels formats of five queries
        run_model3.py : file to run the bm25_model, will generate file name BM25_Model_Output.txt having Qrels formats of five queries
        run.sh: file by which we can run the module
        Makefile: implemented to run the module
        queries : stores queries as guieded in ques3
        rankedlist : ranklist of queries according to ground truth
        output files => Boolean_Model_Output.txt , tfidf_Model_Output.txt , BM25_Model_Output.txt
   
Structure of Module:
        english-corpora                                                #not included in module rest all are included
        ./bm25_model.py
        ./boolean_retrieval_model.py
        ./files_with_index.dat
        ./Makefile
        ./pg_list.dat
        ./preprocessing.py
        ./queries.txt
        ./rankedlist.txt
        ./run_model1.py
        ./run_model2.py
        ./run_model3.py
        ./run.sh
        ./tf_idf_model.py


How to run:
    Run sequence ==>
        make req                                # must run command will install nltk and required package and take executable access to all files in the folder
        make preprocessing                      # skipable if pg_list and files_with_index files already there , it do preprocessing of all the english-corpora and make pg_list and files_with_index (will take approx 15 minutes)
        make run QUERIES="query_filename"     # to run the queries for ex : we have file named query.txt then to run the module : make run QUERIES=query.txt