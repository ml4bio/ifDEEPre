#!/usr/bin/env python

import sys, getopt
from os import path
import os
import numpy as np
import sys
import pickle
import protein_sequence_process_functions as pfunc
from generate_features import get_funcd, get_esm_1b
MAX_LENGTH = 1250
import pdb
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

feature_folder = ''

def processing_input_parameter(argv):
    inputfile = ''
    outputfile = ''
    index = -1
    try:
        opts, args = getopt.getopt(argv,"hi:o:m:",["ifile=","ofile=","mth="])
#        print(0)
    except getopt.GetoptError:
        print ('pre_processing_first_digit.py -i <inputfile> -o <outputfile> -m <query the nth sequence within the file>')
        sys.exit(2)
    for opt, arg in opts:
#        print(0)
        if opt == '-h':
            print ('pre_processing_first_digit.py -i <inputfile> -o <outputfile> -m <query the nth sequence within the file>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-m", "--mth"):
            index = int(arg)
    print ('Input file is ', inputfile)
    print ('Output file is ', outputfile)
    print ('Total index of sequence is ',index)
    return inputfile,outputfile, index


def line2flag(line):
    if '>Yes' in line:
        return 1
    else:
        return 0


def inputfile_processing(inputfile):
    '''
	We would like to generate a dictionary, with the sequence being the key
	and the description of the sequence being the value. We would like to
	print the description in the final output file as well.
    we would like to generate a sure flag list as well to indicate whether
    the user is sure about the input is an enzyme or not.
    '''
    with open(inputfile, 'r') as f:
        text = f.read()
        line_list = text.splitlines()
    line_list = list(filter(None, line_list))
    sequence_dict = dict(zip(line_list[1::2], line_list[0::2]))
    sequence_list = filter(lambda x: '>' not in x, line_list)
    flag_line_list = filter(lambda x: '>' in x, line_list)
    sure_flag_list = map(line2flag,flag_line_list)
    return sequence_list, sequence_dict, sure_flag_list


def set_feature_folder(inputfile):
    global feature_folder
    filename = path.splitext(path.basename(inputfile))[0]
    feature_folder = '../tmp/'+filename


def create_feature_folder(inputfile):
    '''
    check whether the feature folder has already been there
    if not, create the feature folder
    if it has already been existed, warning that we would 
    overwrite the feature.
    '''
    global feature_folder
    if path.isdir(feature_folder):
        print('Warning: the feature folder has already existed, we would overwrite the content!!')
    else:
        os.mkdir(feature_folder)


def concate_digit(first_list, second_list):
    zipped_list = zip(first_list, second_list)
    return map(lambda x: x[0] + '.' + x[1], zipped_list)


def get_features(sequence_list, ID_sequences_with_predictions_esm):
    global feature_folder
    funcd = get_funcd(sequence_list, feature_folder)
    esm_1b = get_esm_1b(sequence_list, feature_folder, ID_sequences_with_predictions_esm)
    return funcd, esm_1b


def check_length(sequence):
    length = len(sequence)
    if length > (1024-2) or length < 50:
        error_infor = '''
        The input does satisfied the lenght requirement!!
        Please only input sequence with length from 50 AA to (1024-2) AA!!
        '''
        raise Exception(error_infor)


def check_sequence_length(sequence_list):
    map(check_length, sequence_list)


def feature_post_processing(pssm, acc, ss):
    pssm = list(map(lambda x: pfunc.feature_length_extend(x, MAX_LENGTH), pssm))
    acc = list(map(lambda x: pfunc.feature_length_extend(x, MAX_LENGTH), acc))
    ss = list(map(lambda x: pfunc.feature_length_extend(x, MAX_LENGTH), ss))
    pssm = np.array(pssm)
    acc = np.array(acc)
    ss = np.array(ss)
    return pssm, acc, ss


def check_and_mkdir(folder):
    if path.isdir(folder):
        print('Warning: the folder already exist, we would overwrite the content')
    else:
        os.mkdir(folder)
    return folder


def save_feature(funcd, esm_1b, output_folder):
    global feature_folder
    if output_folder == '':
        output_folder = feature_folder + '/feature_and_result'
    check_and_mkdir(output_folder)

    with open(output_folder + '/funcd_esm1b.pickle','wb') as f:
        pickle.dump([funcd, esm_1b], f)
    return output_folder


def save_first_digit_result(sequence_list, result_list, output_folder):
    with open(output_folder +'/first_digit_result.txt', 'w') as f:
        for i in range(len(sequence_list)):
            f.write('>'+str(result_list[i])+'\n'+sequence_list[i]+'\n')


def save_first_digit_result_single(sequence, result, output_folder):
    with open(output_folder +'/first_digit_result.txt', 'a') as f:
        f.write('>'+str(result)+'\n'+sequence+'\n')


def save_prob_result_single(sequence, result, output_folder, index, 
    flag, pred_result):
    result = list(result[0])
    result = map(str, result)
    labels = ['0','1','2','3','4','5', '6']
    with open(output_folder+'/log.txt', 'a') as f:
        f.write('The {}th sequence: {}\n'.format(index,sequence))
        f.write('The first digit result:\n')
        f.write('Predicted digit:\n')
        f.write('{}\n'.format(pred_result))
        f.write('Candiate digits:\n')
        if flag:
            f.write('\t'.join(labels[1:])+'\n')
        else:
            f.write('\t'.join(labels)+'\n')
        f.write('Probability:\n')
        f.write('\t'.join(result)+'\n')


## new
class StrToBytes:
    def __init__(self, filestr):
        self.filestr = filestr
    def read(self,size):
        return self.filestr.read(size).encode()
    def readline(self, size=-1):
        return self.filestr.readline(size).encode()



def main_pre_processing(inputfile, outputfile, index):
    print('We are going to prepare all the needed features.')
    set_feature_folder(inputfile)
    # Probably we would not save the feature files, we would keep it in memory
    # But we would put the tmp file of blast and pfam into the folder
    sequence_list, sequence_dict, sure_flag_list = inputfile_processing(inputfile)
    # check sequence length, we accept sequence lenght with the range of 50 to 1250
    check_sequence_length(sequence_list)
    
    ## transform 'map' object to list
    sure_flag_list = list(sure_flag_list)
    sequence_list = list(sequence_list)
    
    ## read the list that store the IDs of sequences that we can model so as to get the np array of esm features in right order (because some are deleted)
    filename_1 = path.splitext(path.basename(inputfile))[0]
    ID_sequences_with_predictions_esm = pickle.load(open('../test_sequence/'+filename_1+'_fasta_ID_sequences_with_predictions.p', "rb"))
    
    ##################################### get features
    create_feature_folder(inputfile)
    funcd, esm_1b = get_features(sequence_list, ID_sequences_with_predictions_esm)
    output_folder = save_feature(funcd, esm_1b, outputfile)




# original
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print(sys.argv[1:])
    inputfile, outputfile, index = processing_input_parameter(sys.argv[1:])
    main_pre_processing(inputfile, outputfile, index)



 
# # directly run the code--example    
 
# ## new  
# #if __name__ == "__main__":
    
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # print(sys.argv[1:])

# # target_path = "../test_sequence/case_study.fasta"
# # target_path = '-i', '../test_sequence/case_study.fasta'
# # target_path = '-i', '../test_sequence/case_study.fasta', "-m", "0"
# # target_path = '-i', '../test_sequence/case_study_bk.fasta', "-m", "0"
# # target_path = '-i', '../test_sequence/sample_input_processed.fasta', "-m", "0"
# target_path = '-i', '../test_sequence/rrrr_rrr_rr_Lbin26_processed.fasta', "-m", "0"

# # inputfile, outputfile, index = processing_input_parameter(sys.argv[1:])
# # inputfile, outputfile, index = processing_input_parameter(target_path)
# inputfile, outputfile, index = processing_input_parameter(target_path)

# main_pre_processing(inputfile, outputfile, index)
















