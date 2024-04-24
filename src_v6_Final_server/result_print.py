#!/usr/bin/env python

import sys, getopt
from os import path
import os
import pickle


def processing_input_parameter(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print ('result_print.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('result_print.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    print ('Input file is ', inputfile)
    return inputfile


def get_output_folder(inputfile):
    filename = path.splitext(path.basename(inputfile))[0]
    output_folder = '../tmp/'+filename+'/feature_and_result'
    return output_folder


def first_digit_post_processing(file):
    with open(file, 'r') as f:
        text = f.read()
    text_list = text.splitlines()
    label_list = filter(lambda x: '>' in x, text_list)
    label_list = list(map(lambda x: x.strip('>'), label_list))
    return label_list


def second_digit_post_processing(file):
    with open(file, 'r') as f:
        text = f.read()
    text_list = text.splitlines()
    label_list = list(map(lambda x: x.split('is')[1], text_list))
    label_list = list(map(lambda x: x.strip(), label_list))
    label_list = list(map(lambda x: x.strip(']'), label_list))
    label_list = list(map(lambda x: x.strip('['), label_list))
    return label_list


def get_sequence_list(file):
    with open(file, 'r') as f:
        text = f.read()
    text_list = text.splitlines()
    sequence_list = list(filter(lambda x: '>' not in x, text_list))
    return sequence_list


def check_pssm(feature_folder):
    path.isfile()


def main_fourth(inputfile):
    output_folder = get_output_folder(inputfile)

    if path.isfile(output_folder+'/funcd_esm1b.pickle'):

        with open(output_folder+'/funcd_esm1b.pickle', 'rb') as f:
            funcd, esm_1b = pickle.load(f, encoding='iso-8859-1')

        for kk in range(len(funcd)):
            result_output_folder = output_folder+'/'+str(kk)
            first_label_list = first_digit_post_processing(result_output_folder+'/first_digit_result.txt')
            second_label_list = second_digit_post_processing(result_output_folder+'/second_digit_result.txt')
            third_label_list = second_digit_post_processing(result_output_folder+'/third_digit_result.txt')
            fourth_label_list = second_digit_post_processing(result_output_folder+'/fourth_digit_result.txt')
            sequence_list = get_sequence_list(result_output_folder+'/first_digit_result.txt')
            
            final_label_list = list(map(lambda x: first_label_list[x]+'.'+second_label_list[x]+'.'+third_label_list[x]+'.-', 
                range(len(list(first_label_list)))))
            
            four_labels_list = list(map(lambda x: first_label_list[x]+'.'+second_label_list[x]+
                '.'+third_label_list[x]+'.'+fourth_label_list[x], 
                range(len(list(first_label_list)))))
            
            with open('ec_mapping_3_digits.csv','r') as f:
                text = f.read()
                mapping_list = text.splitlines()
            mapping_dict=dict()
            for i in range(len(mapping_list)):
                mapping_dict[mapping_list[i].split(',')[0]]=mapping_list[i]
            print('This is the final result!!')
            print(kk)

            for i in range(len(list(sequence_list))):
                print(list(sequence_list))
                print((list(sequence_list))[i]+','+mapping_dict[final_label_list[i]]+','+four_labels_list[i])
    else:
        print('This is the final error message!!')
        print('Feature files not generated!!!')
        print('If you are inputting file, please first check whether the input file fulfills the format requirement!')
        print('If you are inputting a single sequence, that means PSI-BLAST does not find any hits against swissprot for the input sequence!')
        print('Our server can run over this extreme case. If you would like to do that, please contact the developer.')






# ## original

if __name__ == '__main__':
     inputfile = processing_input_parameter(sys.argv[1:])
     main_fourth(inputfile)

# code for running the above code:
    # python result_print.py -i ../test_sequence/case_study_bk.fasta




# # directly run the code--example  

# ## new  
# #if __name__ == "__main__":
    
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # print(sys.argv[1:])

# # target_path = "../test_sequence/case_study.fasta"
# # target_path = '-i', '../test_sequence/case_study.fasta'
# # target_path = '-i', '../test_sequence/case_study_bk.fasta'
# target_path = '-i', '../test_sequence/case_study_bk_processed.fasta'

# inputfile = processing_input_parameter(target_path)

# main_fourth(inputfile)










