#!/usr/bin/env python

import sys, getopt
from os import path
import os
import numpy as np
import sys
#import cPickle
import pickle
import pdb


## check how many sequences in the fasta file
def line2flag(line):
   if '>Yes' in line:
      return 1
   else:
      return 0


def inputfile_processing(inputfile):
   with open(inputfile, 'r') as f:
      text = f.read()
      line_list = text.splitlines()
   line_list = list(filter(None, line_list))
   sequence_dict = dict(zip(line_list[1::2], line_list[0::2]))
   sequence_list = filter(lambda x: '>' not in x, line_list)
   flag_line_list = filter(lambda x: '>' in x, line_list)
   sure_flag_list = list(map(line2flag,flag_line_list))
   return sequence_list, sequence_dict, sure_flag_list


def function_record_row_num(inputfile):
   with open(inputfile, 'r') as f:
      text = f.read()
      line_list_a = text.splitlines()
   line_list_a = list(filter(None, line_list_a))
   row_num_list = []
   for i in range(len(line_list_a)):
       if '>'in line_list_a[i]:
           row_num_list.append(i)
   return row_num_list, line_list_a


def delete_specific_element(sequence_list,delete_index):
    for i in delete_index:
        sequence_list.pop(i)


def output_fasta_file(sequence,path,filename,i,mark):
    if mark ==0:
        f=open(path+filename,'w')
        f.write('>'+str(i)+'\n')
        f.write(sequence)
        f.close()
    else:
        f=open(path+filename,'a')
        f.write('\n>'+str(i)+'\n')
        f.write(sequence)
        f.close()


def check_and_mkdir(folder):
    if path.isdir(folder):
        print('Warning: the folder already exist, we would overwrite the content')
    else:
        os.mkdir(folder)
    return folder



####################### process the data: (delete sequences longer than 1024 or shorter than 50) and saved the processed data
####################### process the data: (delete sequences longer than 1024 or shorter than 50) and saved the processed data
####################### process the data: (delete sequences longer than 1024 or shorter than 50) and saved the processed data

print('processing the input sequence data')

filename_before_process = 'ifdeepre_inputExample'
inputfile_before_process = '../test_sequence/'+filename_before_process+'.fasta'


############################################################ mending data from uniport
sequence_list_before_process_1, sequence_dict_before_process, sure_flag_list_before_process = inputfile_processing(inputfile_before_process)


## the lines of the downloaded txt files is totally in mess, we write a function to mend it
record_row_num_with_mark, line_list_a_mess = function_record_row_num(inputfile_before_process)
record_row_num_with_mark = np.array(record_row_num_with_mark)

append_num_of_rows = []
for kk in range(len(record_row_num_with_mark)):
    if (kk >=1) and (kk <= len(record_row_num_with_mark)-1):
            num_of_rows_current = record_row_num_with_mark[kk] - record_row_num_with_mark[kk-1] - 1
            append_num_of_rows.append(num_of_rows_current)

num_of_rows_current = len(line_list_a_mess) - record_row_num_with_mark[-1] - 1
append_num_of_rows.append(num_of_rows_current)

append_num_of_rows = np.array(append_num_of_rows)

## transform 'map' object to list
sequence_list_before_process_1 = list(sequence_list_before_process_1)


## get the real sequences:
sequence_list_before_process = []
current_row = 0
for qq in range(len(append_num_of_rows)-1):
# for qq in range(5):
    for jj in range(append_num_of_rows[qq]):
        if jj == 0:
            current_list = sequence_list_before_process_1[current_row+jj]
        if jj > 0:
            current_list = current_list + sequence_list_before_process_1[current_row+jj]
    sequence_list_before_process.append(current_list)
    current_row = current_row + append_num_of_rows[qq]
    # print("current_row")
    # print(current_row)


## append from sequence_list_before_process_1[current_row] to last
for pp in range(len(sequence_list_before_process_1)-current_row):
    # print(pp)
    if pp == 0:
        current_list = sequence_list_before_process_1[current_row+pp]
#    if jj > 0:
    if pp > 0:
        current_list = current_list + sequence_list_before_process_1[current_row+pp]
sequence_list_before_process.append(current_list)
############################################################ mending data from uniport



## build a list of array to store the ids of sequences longer than 1024 or shorter than 50
ID_longer_than_1024_or_shorter_than_50 = list([])

## build a list to store the IDs of sequences that we can analyze (will be used later)
ID_sequences_with_predictions = list([])

mark = 0
#output the sequence into files according to their order in the list, start from 0
for i in range(len(sequence_list_before_process)):
    if i%1000==0:
        print(i)
    filename_processing = filename_before_process+'_processed.fasta'
    if (len(sequence_list_before_process[i]) > (1024-2)) or (len(sequence_list_before_process[i]) < 50):
        ID_longer_than_1024_or_shorter_than_50.append(i)
    else:
        ID_sequences_with_predictions.append(i)
        output_fasta_file(sequence_list_before_process[i],'../test_sequence/',filename_processing,i, mark)
        mark = 1


ID_longer_than_1024_or_shorter_than_50 = np.array(ID_longer_than_1024_or_shorter_than_50)

pickle.dump(ID_longer_than_1024_or_shorter_than_50, open('../test_sequence/'+filename_before_process+'_fasta_ID_longer_than_1024_or_shorter_than_50.p', 'wb'))

pickle.dump(ID_sequences_with_predictions, open('../test_sequence/'+filename_before_process+'_processed_fasta_ID_sequences_with_predictions.p', 'wb'))








######################################################### get funcd and esm features for the processed data
######################################################### get funcd and esm features for the processed data
######################################################### get funcd and esm features for the processed data

print('extracting representations')

filename = filename_before_process+'_processed'
inputfile = '../test_sequence/'+filename+'.fasta'


######################################################### firstly, we get esm features ready, namely .pt files ready
######################################################### firstly, we get esm features ready, namely .pt files ready
cmd1 = 'python extract_esm_1b.py esm1b_t33_650M_UR50S ' +inputfile + ' ../tmp/'+filename+'/esm_1b_output' +' --repr_layers 33 --include mean'
os.environ['MKL_THREADING_LAYER']='GNU'
print(cmd1)
os.system(cmd1)



############################################## secondly, we getfuncd features, and transform esm .pt files to np aray
############################################## secondly, we getfuncd features, and transform esm .pt files to np aray
sequence_list, sequence_dict, sure_flag_list = inputfile_processing(inputfile)

cmd1 = 'python code_0_pre_processing.py -i ../test_sequence/'+filename+'.fasta -m '+str(0)
print(cmd1)
os.system(cmd1)








############################################################################################# perform predictions
############################################################################################### perform predictions
############################################################################################### perform predictions

seq_num = len(sure_flag_list)


####################################################################### predict first digit
####################################################################### predict first digit
########### perform prediction once for all

print('predicting first EC number digit')

cmd1 = 'python code_1_first_digit.py -i ../test_sequence/'+filename+'.fasta -m '+str(0)
print(cmd1)
os.system(cmd1)




################################################################### predict second digit
#################################################################### predict second digit
# ########### perform prediction for each main class one by one: within each class, 
# we can perfom batch prediction: class 0, 1, 2, 3, 4, 5, 6

print('predicting second EC number digit')

for i in range(7):
    # print ("We are going to use the %dth main class model to perform prediction"%i)
    cmd2 = 'python code_2_second_digit_post_processing.py -i ../test_sequence/'+filename+'.fasta -m 0 -c '+str(i)
    # print(cmd2)
    os.system(cmd2)




##################################################################predict third and forth digit
################################################################## predict third and forth digit

print('predicting third and forth EC number digit')


## the 0th type of main class
i=0
## the jth type of sub-class
print ("predict third and forth digit of %dth main class"%(i))
for j in [0]:
    # print("i")
    # print(i)
    # print("j")
    # print(j)
    # print ("We are going to use the %dth main class and the %dth sub-class model to perform prediction"%(i, j))
    cmd2 = 'python code_3_third_digit_post_processing.py -i ../test_sequence/'+filename+'.fasta -m 0 -c '+str(i)+' -s '+str(j)
    # print(cmd2)
    os.system(cmd2)




## the 1th type of main class
i=1
## the jth type of sub-class
print ("predict third and forth digit of %dth main class"%(i))
for j in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 21, 15, 20, 97, 9]:
    # print("i")
    # print(i)
    # print("j")
    # print(j)
    # print ("We are going to use the %dth main class and the %dth sub-class model to perform prediction"%(i, j))
    cmd2 = 'python code_3_third_digit_post_processing.py -i ../test_sequence/'+filename+'.fasta -m 0 -c '+str(i)+' -s '+str(j)
    # print(cmd2)
    os.system(cmd2)




## the 2th type of main class
i=2
## the jth type of sub-class
print ("predict third and forth digit of %dth main class"%(i))
for j in [1, 3, 4, 6, 7, 8, 2, 5, 9, 10]:
    # print("i")
    # print(i)
    # print("j")
    # print(j)
    # print ("We are going to use the %dth main class and the %dth sub-class model to perform prediction"%(i, j))
    cmd2 = 'python code_3_third_digit_post_processing.py -i ../test_sequence/'+filename+'.fasta -m 0 -c '+str(i)+' -s '+str(j)
    # print(cmd2)
    os.system(cmd2)



## the 3th type of main class
i=3
## the jth type of sub-class
print ("predict third and forth digit of %dth main class"%(i))
for j in [1, 2, 3, 4, 5, 6, 7, 8, 11]:
    # print("i")
    # print(i)
    # print("j")
    # print(j)
    # print ("We are going to use the %dth main class and the %dth sub-class model to perform prediction"%(i, j))
    cmd2 = 'python code_3_third_digit_post_processing.py -i ../test_sequence/'+filename+'.fasta -m 0 -c '+str(i)+' -s '+str(j)
    # print(cmd2)
    os.system(cmd2)



## the 4th type of main class
i=4
## the jth type of sub-class
print ("predict third and forth digit of %dth main class"%(i))
for j in [1, 2, 3, 4, 6, 99]:
    # print("i")
    # print(i)
    # print("j")
    # print(j)
    # print ("We are going to use the %dth main class and the %dth sub-class model to perform prediction"%(i, j))
    cmd2 = 'python code_3_third_digit_post_processing.py -i ../test_sequence/'+filename+'.fasta -m 0 -c '+str(i)+' -s '+str(j)
    # print(cmd2)
    os.system(cmd2)



## the 5th type of main class
i=5
## the jth type of sub-class
print ("predict third and forth digit of %dth main class"%(i))
for j in [1, 3, 4, 2, 5, 99]:
    # print("i")
    # print(i)
    # print("j")
    # print(j)
    # print ("We are going to use the %dth main class and the %dth sub-class model to perform prediction"%(i, j))
    cmd2 = 'python code_3_third_digit_post_processing.py -i ../test_sequence/'+filename+'.fasta -m 0 -c '+str(i)+' -s '+str(j)
    # print(cmd2)
    os.system(cmd2)



## the 6th type of main class
i=6
## the jth type of sub-class
print ("predict third and forth digit of %dth main class"%(i))
for j in [3, 1, 2, 4, 5, 6]:
    # print("i")
    # print(i)
    # print("j")
    # print(j)
    # print ("We are going to use the %dth main class and the %dth sub-class model to perform prediction"%(i, j))
    cmd2 = 'python code_3_third_digit_post_processing.py -i ../test_sequence/'+filename+'.fasta -m 0 -c '+str(i)+' -s '+str(j)
    # print(cmd2)
    os.system(cmd2)





##################################################################################################### print the final results
##################################################################################################### print the final results
##################################################################################################### print the final results
# python result_print.py -i ${FILENAME}

print('final EC nunber prediction results')

cmd4 = 'python result_print.py -i ../test_sequence/'+filename+'.fasta'
# print(cmd4)
os.system(cmd4)




