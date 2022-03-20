import sys

import tf_idf_model as tf
import pickle


pg_list = pickle.load(open('pg_list.dat','rb'))
files_with_index = pickle.load(open('files_with_index.dat','rb'))



with open(sys.argv[1], 'r', encoding = 'utf8') as f:
#     query_g = f.readlines()
    query_g = [line.rstrip() for line in f]
    query = [query_g[i][4:] for i in range(len(query_g))]

#generating an output file for the system, to store the results in the format of QRels
output = open('tfidf_Model_Output.txt','w+')

for i in range(len(query)):
    if len(query[i]) <= 0:
        continue
    output_files = tf.tf_idf(query[i])
    k = 5
    if len(output_files) <= k:
        if i < 10:
            for j in range(len(output_files)):
                output.write('Q0' + str(i+1) + ', ' + '1' + ', ' + output_files[j] + ', ' + '1' + '\n')
            files = [file for file in files_with_index.values() if file not in output_files]
            j = len(output_files)
            for file in files[0:k - j]:
                output.write('Q0' + str(i+1) + ', ' + '1' + ', ' + file + ', ' + '0' + '\n')
                j += 1

        else:
            for j in range(len(output_files)):
                output.write('Q' + str(i+1) + ', ' + '1' + ', ' + output_files[j] + ', ' + '1' + '\n')
            files = [file for file in files_with_index.values() if file not in output_files]
            j = len(output_files)
            for file in files[0:k - j]:
                output.write('Q' + str(i+1) + ', ' + '1' + ', ' + file + ', ' + '0' + '\n')
                j += 1
    else:
        if i < 10:
            for j in range(0,k):
                output.write('Q0' + str(i+1) + ', ' + "1" + ', ' + output_files[j] + ', ' + '1' + '\n')
        else:
            for j in range(0,k):
                output.write('Q' + str(i+1) + ', ' + '1' + ', ' + output_files[j] + ', ' + '1' + '\n')
output.close()
