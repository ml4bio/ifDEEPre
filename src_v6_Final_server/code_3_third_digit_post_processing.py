#!/usr/bin/env python

import sys, getopt
from os import path
import os
from third_num_predict import predict_third_digit
from fourth_num_predict import fourth_num_predict
from code_1_first_digit import inputfile_processing
import numpy as np
import sys
import pickle
import protein_sequence_process_functions as pfunc
from generate_features import get_funcd, get_esm_1b
import pdb
MAX_LENGTH = 1250

feature_folder = ''


def processing_input_parameter_level3(argv):
    inputfile = ''
    outputfile = ''
    index = -1
    level_1_class = 0
    level_2_subclass = 0
    try:
        opts, args = getopt.getopt(argv,"hi:o:m:c:s:",["ifile=","ofile=","mth=","loneth=","ltwoth="])
    except getopt.GetoptError:
        # print ('second_digit_post_processing.py -i <inputfile> -o <outputfile> -m <query the nth sequence within the file> -c <the kth class in level 1> -s <the sth subclass in level 2>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            # print ('second_digit_post_processing.py -i <inputfile> -o <outputfile> -m <query the nth sequence within the file> -c <the kth class in level 1> -s <the sth subclass in level 2>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-m", "--mth"):
            index = int(arg)
        elif opt in ("-c", "--loneth"):
            level_1_class = int(arg)
        elif opt in ("-s", "--ltwoth"):
            level_2_subclass = int(arg)
    # print ('Input file is ', inputfile)
    # print ('Output file is ', outputfile)
    # print ("We are going to predict the %dth query in the file"%index)
    # print ("We are going to use the %dth class in level 1 to perform prediction"%level_1_class)
    # print ("We are going to use the %dth subclass model in level 2 to perform prediction"%level_2_subclass)
    return inputfile, outputfile, index, level_1_class, level_2_subclass


def load_first_digit_result(saved_feature_folder, num_id):
    with open(saved_feature_folder + '/' + str(num_id) + '/first_digit_result.txt') as f:
        text = f.read()
    line_list = text.splitlines()
    label_lines = filter(lambda x: '>' in x, line_list)
    label_list = map(lambda x: int(x.strip('>')), label_lines)
    return label_list


def second_digit_post_processing(file):
    with open(file, 'r') as f:
        text = f.read()
    text_list = text.splitlines()
    label_list = map(lambda x: x.split('is')[1], text_list)
    label_list = map(lambda x: x.strip(), label_list)
    label_list = map(lambda x: x.strip(']'), label_list)
    label_list = map(lambda x: x.strip('['), label_list)
    return label_list


def load_second_digit_result(saved_feature_folder, num_id):
    file_path = saved_feature_folder + '/' + str(num_id) + '/second_digit_result.txt'
    label_list = second_digit_post_processing(file_path)
    return label_list


# a function to append the prediciton result of the third digit to the result file.
def append_result(saved_feature_folder, third_digit_result, num_id, original_id_list):
    result_output_folder = saved_feature_folder +'/'+str(original_id_list[num_id])
    with open(result_output_folder + '/third_digit_result.txt', 'w') as f:
        f.write('The result EC number of the '+str(original_id_list[num_id])+'th sequence is ['+str(third_digit_result)+']\n')


def append_fourth_result(saved_feature_folder, third_digit_result, original_id):
    with open(saved_feature_folder + '/' + str(original_id) + '/fourth_digit_result.txt', 'w') as f:
        f.write('The result EC number of the '+str(original_id)+'th sequence is '+str(third_digit_result)+'\n')


def save_prob_result_single(output_folder, labels, result , num_id, pred_result, original_id_list):
    original_id = original_id_list[num_id]
    result_output_folder = output_folder +'/'+str(original_id)
    result = map(str, result)
    labels = map(str, labels)
    with open(result_output_folder+'/log.txt', 'a') as f:
        f.write('The third digit result:\n')
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
        f.write('The third digit result:\n')
        f.write('Predicted digit:\n')
        # f.write('{}\n'.format(pred_result[0]))
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


def main_third(inputfile, outputfile, index, level_1_class, level_2_subclass):
    sequence_list, _, _ = inputfile_processing(inputfile)
    filename = path.splitext(path.basename(inputfile))[0]
    saved_feature_folder = '../tmp/'+filename+'/feature_and_result'

    ## load saved features
    with open(saved_feature_folder+'/funcd_esm1b.pickle', 'rb') as f:
        funcd, esm_1b = pickle.load(f, encoding='iso-8859-1')
    # print("funcd.shape")
    # print(funcd.shape)

    # preserved
    ## firstly load first_digit_result
    first_digit_result = []
    for kk in range(len(funcd)):
        # print(kk)
        first_digit_result_current = load_first_digit_result(saved_feature_folder, kk)
        ## transform 'map' object to list
        ## the first digit of the kth sequence
        first_digit_result_current = list(first_digit_result_current)
        first_digit_result.append(first_digit_result_current[0])
    ## trenaform from list to np
    first_digit_result = np.array(first_digit_result)
    
    ## then load second_digit_result
    second_digit_result = []
    for kk in range(len(funcd)):
        # print(kk)
        second_digit_result_current = load_second_digit_result(saved_feature_folder, kk)
        ## transform 'map' object to list
        ## the second digit of the kth sequence
        second_digit_result_current = list(second_digit_result_current)
        second_digit_result.append(int(second_digit_result_current[0]))
    ## trenaform from list to np
    second_digit_result = np.array(second_digit_result)


    ############################################################## predict the third digit in batch for the level_1_class main class and the level_2_subclass sub-class
    ############################################################## predict the third digit in batch for the level_1_class main class and the level_2_subclass sub-class
    # print("using the class ["+str(level_1_class)+'] and the sub-class ['+str(level_2_subclass)+'] to predic the third digit')
    # print("using the class ["+str(level_1_class)+'] and the sub-class ['+str(level_2_subclass)+'] to predic the third digit')
    ## record all the ids for storing the prediction results in corresponding folder
    # i indicates the main class and j indicates the sub-class
    id_class_i_j_list = []
    ## fisrt disgits of class [level_1_class]
    first_digit_result_class_i_j_list = []
    ## second disgits of class [level_2_subclass]
    second_digit_result_class_i_j_list = []
    ## funcd of class [i]
    funcd_class_i_j_list = []
    ## funcd of class [i]
    esm_1b_class_i_j_list = []
    
    for qq in range(len(first_digit_result)):
        if (first_digit_result[qq] == level_1_class) and (second_digit_result[qq] == level_2_subclass):
            # print("first_digit_result[qq]")
            # print(first_digit_result[qq])
            # print("second_digit_result[qq]")
            # print(second_digit_result[qq])
            ## record all the ids of class [i]
            id_class_i_j_list.append(qq)
            ## record first disgits of class [i]
            first_digit_result_class_i_j_list.append(first_digit_result[qq])
            ## record second disgits of subclass [j]
            second_digit_result_class_i_j_list.append(second_digit_result[qq])
            ## record funcd of class [i]
            funcd_class_i_j_list.append(funcd[qq])
            ## record esm_1b of class [i]
            esm_1b_class_i_j_list.append(esm_1b[qq])
    
    ################### if this class have sequences
    ## transform the data from list to np array
    if len(id_class_i_j_list) > 0:
        ## record fisrt disgits of class [i]
        first_digit_result_class_i_j = np.array(first_digit_result_class_i_j_list)
        ## record second disgits of sub-class [j]
        second_digit_result_class_i_j = np.array(second_digit_result_class_i_j_list)
        ## record funcd of class [i]
        funcd_class_i_j = np.array(funcd_class_i_j_list)
        ## record esm_1b of class [i]
        esm_1b_class_i_j = np.array(esm_1b_class_i_j_list)
        third_digit_result_class_i_j,labels_class_i_j,prob_list_class_i_j=predict_third_digit(funcd_class_i_j,esm_1b_class_i_j,first_digit_result_class_i_j,second_digit_result_class_i_j)
        
        ## test whether it works
        print("third_digit_result_class_i_j")
        print(third_digit_result_class_i_j)
        print("labels_class_i_j")
        print(labels_class_i_j)
        print("prob_list_class_i_j")
        print(prob_list_class_i_j)
        
        # save results to txt files
        if (len(third_digit_result_class_i_j) > 1):
            for ii in range(len(third_digit_result_class_i_j)):
                save_prob_result_more(saved_feature_folder, labels_class_i_j, prob_list_class_i_j[ii], ii, third_digit_result_class_i_j[ii], id_class_i_j_list)
                
        if (len(third_digit_result_class_i_j) == 1):
            save_prob_result_single(saved_feature_folder, labels_class_i_j, prob_list_class_i_j, 0, third_digit_result_class_i_j, id_class_i_j_list)
            
        for jj in range(len(third_digit_result_class_i_j)):
            try:
                # print("try")
                append_result(saved_feature_folder, third_digit_result_class_i_j[jj], jj, id_class_i_j_list)
            except Exception as e:
                print("except Exception as e")
                print(e)
                append_result(saved_feature_folder, -1, jj, id_class_i_j_list)








    ############################################################################### predict the forth digit
    ############################################################################### predict the forth digit

    ## new, we predict all the sequences in the mian class [level_1_class] and the sub-class [level_2_subclass]
    # print ("We are predicting the fourth digit of the %dth main class and the %dth sub-class"%(level_1_class, level_2_subclass))
    
    ## transform 'filter' object to list
    sequence_list = list(sequence_list) 
    
    for pp in range(len(id_class_i_j_list)):
        ## store the foyrth digit
        fourth_digit_current = list()
        ## original id of the sequence
        original_id_num = id_class_i_j_list[pp]
        # print("original_id_num")
        # print(original_id_num)
        ## predict the fourth digit
        # print("current sequence_list")
        # print(sequence_list[original_id_num])
        # fourth_digit_current.append(fourth_num_predict(saved_feature_folder, sequence_list[original_id_num]))
        fourth_digit_current.append(fourth_num_predict(saved_feature_folder, sequence_list[original_id_num], level_1_class))
        # print("predicted fourth_digit")
        # print(fourth_digit_current)
        
        ## save the results to txt file
        append_fourth_result(saved_feature_folder, fourth_digit_current, original_id_num)






# ## original

if __name__ == "__main__":
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    ##################### new
      print(sys.argv[1:])
    ##################### new    
      # inputfile, outputfile, index = processing_input_parameter(sys.argv[1:])
      inputfile, outputfile, index, level_1_class, level_2_subclass = processing_input_parameter_level3(sys.argv[1:])
      main_third(inputfile, outputfile, index, level_1_class, level_2_subclass)


# # code for running the above code:
#     # python code_3_third_digit_post_processing.py -i ../test_sequence/case_study.fasta
#     # python code_3_third_digit_post_processing.py -i ../test_sequence/case_study_bk.fasta -m 0





# # directly run the code--example    

# ## new  
# #if __name__ == "__main__":
    
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # print(sys.argv[1:])

# # target_path = "../test_sequence/case_study.fasta"
# # target_path = '-i', '../test_sequence/case_study_bk.fasta', "-m", "0"
# # target_path = '-i', '../test_sequence/case_study_bk.fasta', "-m", "0", "-c", "2", "-s", "7"
# target_path = '-i', '../test_sequence/case_study_bk_processed.fasta', "-m", "0", "-c", "2", "-s", "7"
# target_path = '-i ../test_sequence/case_study_bk_processed.fasta -m 0 -c 1 -s 9'

# # inputfile, outputfile, index = processing_input_parameter(sys.argv[1:])
# inputfile, outputfile, index, level_1_class, level_2_subclass = processing_input_parameter_level3(target_path)

# main_third(inputfile, outputfile, index, level_1_class, level_2_subclass)




















