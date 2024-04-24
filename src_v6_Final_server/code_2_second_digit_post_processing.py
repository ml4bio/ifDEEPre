#!/usr/bin/env python

import sys, getopt
from os import path
import os
from second_num_predict import predict_second_digit
import numpy as np
import sys
import pickle
import protein_sequence_process_functions as pfunc
import pdb
MAX_LENGTH = 1250

feature_folder = ''


def processing_input_parameter_level2(argv):
    inputfile = ''
    outputfile = ''
    index = -1
    level_1_class = 0
    try:
        opts, args = getopt.getopt(argv,"hi:o:m:c:",["ifile=","ofile=","mth=","loneth="])
    except getopt.GetoptError:
        # print ('second_digit_post_processing.py -i <inputfile> -o <outputfile> -m <query the nth sequence within the file> -c <the kth class model in level 1>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            # print ('second_digit_post_processing.py -i <inputfile> -o <outputfile> -m <query the nth sequence within the file> -c <the kth class model in level 1>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-m", "--mth"):
            index = int(arg)
        elif opt in ("-c", "--loneth"):
            level_1_class = int(arg)
    # print ('Input file is ', inputfile)
    # print ('Output file is ', outputfile)
    # print ("We are going to predict the %dth query in the file"%index)
    # print ("We are going to use the %dth class model in level 1 to perform prediction"%level_1_class)
    return inputfile, outputfile, index, level_1_class


def load_first_digit_result(saved_feature_folder, num_id):
    with open(saved_feature_folder + '/' + str(num_id) + '/first_digit_result.txt') as f:
        text = f.read()
    line_list = text.splitlines()
    label_lines = filter(lambda x: '>' in x, line_list)
    label_list = map(lambda x: int(x.strip('>')), label_lines)
    return label_list


# a function to append the prediciton result of the second digit to the result file.
def append_result(saved_feature_folder, second_digit_result, num_id, original_id_list):
    result_output_folder = saved_feature_folder +'/'+str(original_id_list[num_id])
    with open(result_output_folder + '/second_digit_result.txt', 'w') as f:
        f.write('The result EC number of the '+str(original_id_list[num_id])+'th sequence is ['+str(second_digit_result)+']\n')


def save_prob_result_single(output_folder, labels, result , num_id, pred_result, original_id_list):
    original_id = original_id_list[num_id]
    result_output_folder = output_folder +'/'+str(original_id)
    result = map(str, result)
    labels = map(str, labels)
    with open(result_output_folder+'/log.txt', 'a') as f:
        f.write('The second digit result:\n')
        f.write('Predicted digit:\n')
        f.write('{}\n'.format(pred_result[0]))
        f.write('Candidate digits:\n')
        f.write('\t'.join(labels)+'\n')
        f.write('Probability:\n')
        f.write('\t'.join(result)+'\n')


def save_prob_result_more(output_folder, labels, result, num_id, pred_result, original_id_list):
    # print("result")
    # print(result)
    # print("labels")
    # print(labels)
    result = map(str, result)
    labels = map(str, labels)
    # print("result")
    # print(result)
    # print("labels")
    # print(labels)
    original_id = original_id_list[num_id]
    # print("original_id")
    # print(original_id)
    result_output_folder = output_folder +'/'+str(original_id)
    with open(result_output_folder+'/log.txt', 'a') as f:
        f.write('The second digit result:\n')
        f.write('Predicted digit:\n')
        f.write('{}\n'.format(pred_result))
        f.write('Candidate digits:\n')
        f.write('\t'.join(labels)+'\n')
        f.write('Probability:\n')
        f.write('\t'.join(result)+'\n')


class StrToBytes:
    def __init__(self, filestr):
        self.filestr = filestr
    def read(self,size):
        return self.filestr.read(size).encode()
    def readline(self, size=-1):
        return self.filestr.readline(size).encode()


def main_second(inputfile, outputfile, index, level_1_class):
    filename = path.splitext(path.basename(inputfile))[0]
    saved_feature_folder = '../tmp/'+filename+'/feature_and_result'

    ## load saved features
    with open(saved_feature_folder+'/funcd_esm1b.pickle', 'rb') as f:
        funcd, esm_1b = pickle.load(f, encoding='iso-8859-1')

    # load preserved first digit prediction results
    first_digit_result = []
    for kk in range(len(funcd)):
        first_digit_result_current = load_first_digit_result(saved_feature_folder, kk)
        ## transform 'map' object to list
        ## the first digit of the kth sequence
        first_digit_result_current = list(first_digit_result_current)
        first_digit_result.append(first_digit_result_current[0])
    ## trenaform from list to np
    first_digit_result = np.array(first_digit_result)


    ############################################################################################## perform prediction for the level_1_class th class
    # print("perform prediction for class ["+str(level_1_class)+']')
    # print("perform prediction for class ["+str(level_1_class)+']')
    ## record all the ids for storing the prediction results in corresponding folder
    id_class_i_list = []
    ## fisrt disgits of class [level_1_class]
    first_digit_result_class_i_list = []
    ## funcd of class [i]
    funcd_class_i_list = []
    ## funcd of class [i]
    esm_1b_class_i_list = []
    
    for qq in range(len(first_digit_result)):
        if first_digit_result[qq] == level_1_class:
            ## record all the ids of class [i]
            id_class_i_list.append(qq)
            ## record fisrt disgits of class [i]
            first_digit_result_class_i_list.append(first_digit_result[qq])
            ## record funcd of class [i]
            funcd_class_i_list.append(funcd[qq])
            ## record esm_1b of class [i]
            esm_1b_class_i_list.append(esm_1b[qq])
    
    ################### if this class have sequences
    ## transform the data from list to np array
    if len(id_class_i_list) > 0:
        ## record fisrt disgits of class [i]
        first_digit_result_class_i = np.array(first_digit_result_class_i_list)
        ## record funcd of class [i]
        funcd_class_i = np.array(funcd_class_i_list)
        ## record esm_1b of class [i]
        esm_1b_class_i = np.array(esm_1b_class_i_list)
        # ## original
        # second_digit_result, labels, prob_list = predict_second_digit(funcd[index], esm_1b[index], first_digit_result[index])
        ## new
        second_digit_result_class_i, labels_class_i, prob_list_class_i = predict_second_digit(funcd_class_i, esm_1b_class_i, first_digit_result_class_i)
        
        ## test whether it works
        print("second_digit_result_class_i")
        print(second_digit_result_class_i)
        print("labels_class_i")
        print(labels_class_i)
        print("prob_list_class_i")
        print(prob_list_class_i)
        
        # save prediction results
        if (len(second_digit_result_class_i) > 1):
            for ii in range(len(second_digit_result_class_i)):
                save_prob_result_more(saved_feature_folder, labels_class_i, prob_list_class_i[ii], ii, second_digit_result_class_i[ii], id_class_i_list)
                
        if (len(second_digit_result_class_i) == 1):
            save_prob_result_single(saved_feature_folder, labels_class_i, prob_list_class_i, 0, second_digit_result_class_i, id_class_i_list)
            
        for jj in range(len(second_digit_result_class_i)):
            try:
                # print("try")
                append_result(saved_feature_folder, second_digit_result_class_i[jj], jj, id_class_i_list)
            except Exception as e:
                print("except Exception as e")
                print(e)
                append_result(saved_feature_folder, -1, jj, id_class_i_list)








if __name__ == "__main__":
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
      inputfile, outputfile, index, level_1_class = processing_input_parameter_level2(sys.argv[1:])
      main_second(inputfile, outputfile, index, level_1_class)



# # code for running the above code:
#     # python code_2_second_digit_post_processing.py -i ../test_sequence/case_study.fasta
#     # python code_2_second_digit_post_processing.py -i ../test_sequence/case_study_bk.fasta
#     # python code_2_second_digit_post_processing.py -i ../test_sequence/case_study_bk.fasta -m 0









# # directly run the code--example    
 
# ## new  
# #if __name__ == "__main__":
    
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # print(sys.argv[1:])

# # target_path = "../test_sequence/case_study.fasta"
# # target_path = '-i', '../test_sequence/case_study.fasta'
# # target_path = '-i', '../test_sequence/case_study.fasta', "-m", "0"
# # target_path = '-i', '../test_sequence/case_study_bk.fasta', "-m", "0"
# # target_path = '-i', '../test_sequence/case_study_bk_processed.fasta', "-m", "0"
# target_path = '-i', '../test_sequence/case_study_bk_processed.fasta', "-m", "0", "-c", "2"


# # inputfile, outputfile, index = processing_input_parameter(sys.argv[1:])
# # inputfile, outputfile, index = processing_input_parameter(target_path)
# # inputfile, outputfile, index = processing_input_parameter_level2(target_path)
# inputfile, outputfile, index, level_1_class = processing_input_parameter_level2(target_path)

# main_second(inputfile, outputfile, index, level_1_class)





















