#!/usr/bin/env python
import os
from os import path
import numpy as np
import protein_sequence_process_functions as pfunc
import torch


MAX_LENGTH = 1250


def check_and_mkdir(folder):
	if path.isdir(folder):
		print('Warning: the folder already exist, we would overwrite the content')
	else:
		os.mkdir(folder)
	return folder


def output_fasta_file(sequence_list, i, path):
    with open(path+'/'+str(i)+'.fasta','w') as f:
    	f.write('>'+str(i)+'\n')
    	f.write(sequence_list[i])


def get_fasta_files(sequence_list, output_folder1):
    for x in range(len(sequence_list)):
    	output_fasta_file(sequence_list, x, output_folder1)



# The following functions are for blast feature generation and processing

def line_string_to_float_array(line):
	'''
	Function to process each line of a blast output file
	'''
	line_list=line.split()
	first_vector=np.array(list(map(float,line_list[2:22])))
	second_vector=np.array(list(map(float,line_list[22:42])))
	return (first_vector,second_vector)


def file_to_matrix(path,file_name):
	'''
	Function to process each blast output file
	The output are two numpy array. each has dimentionality of length_of_sequence*20
	'''
	f=open(path+file_name,'r')
	file_content=f.read()
	f.close()
	content_list=file_content.splitlines()
	first_matrix,second_matrix=line_string_to_float_array(content_list[3])
	for i in range(4,len(content_list)-6):
	    first_array_temp,second_array_temp=line_string_to_float_array(content_list[i])
	    first_matrix=np.vstack((first_matrix,first_array_temp))
	    second_matrix=np.vstack((second_matrix,second_array_temp))
	return first_matrix


def blast_post_processing(output_folder, file_num):
	list_of_first_matrix = list(map(lambda x: file_to_matrix(output_folder+'/', 
		str(x)+'.ascii_pssm'), list(range(file_num))))
	return list_of_first_matrix


def get_pssm(sequence_list, feature_folder):
	'''
	this function would output the pssm list for the input sequence
	'''
	input_folder = check_and_mkdir(feature_folder+'/blast_input')
	output_folder = check_and_mkdir(feature_folder+'/blast_output')
	get_fasta_files(sequence_list, input_folder)
	cmd = 'bash blast.sh -s '+str(0)+' -e '+str(len(sequence_list)-1)+' -f '+feature_folder
	print('We are going to run blast on those sequence.')
	os.system(cmd)
	pssm_list = blast_post_processing(output_folder, len(sequence_list))

	return pssm_list




# The following functions are for the funcd feature generations and processing
    
def run_hmmscan(input_folder, output_folder, id):
	cmd = 'hmmscan ../database/pfam/Pfam-A.hmm '+input_folder+'/'+str(id)+'.fasta'
	cmd = cmd + ' > '+output_folder+'/'+str(id)+'.out'
	print('We are inside the hmmsanc function!!')
	os.system(cmd)


#start_index and end_index are all included.
def get_names(result,name_start_index,name_end_index):
    name_list=[]
    for i in range(name_start_index,name_end_index+1):
        line_temp=result[i]
        line_list=line_temp.split()
        name_list.append(line_list[8])
    return name_list


def encoding_Pfam_result(path, filename, model_names_list):
	try:
		f=open(path+filename,'r')
		result=f.read()
		f.close()
		result=result.split('\n')
		encoding=np.zeros(16306)
		hit_name_list=[]
		if '   [No hits detected that satisfy reporting thresholds]' not in result:
		    name_end_index=result.index('Domain annotation for each model (and alignments):')-3
		    if '  ------ inclusion threshold ------' in result:
		        name_end_index=result.index('  ------ inclusion threshold ------')-1
		    hit_name_list=get_names(result,14,name_end_index)
		    for name in hit_name_list:
		        index=model_names_list.index(name)
		        encoding[index]=1
		return encoding
	except Exception as e:
		return np.zeros(16306)


def pfam_post_processing(output_folder, file_num):
	with open('../database/pfam/list_of_model_names.txt','r') as f:
		model_names_list=f.read()
	model_names_list=model_names_list.splitlines()
	model_names_list = list(map(lambda x: x[6:], model_names_list))
	funcd_list = list(map(lambda x: encoding_Pfam_result(output_folder+'/',
		str(x)+'.out', model_names_list), range(file_num)))
	return funcd_list


def get_funcd(sequence_list, feature_folder):
	'''
	this function would output the funcd list for the input sequence
	'''
	input_folder = check_and_mkdir(feature_folder+'/pfam_input')
	output_folder = check_and_mkdir(feature_folder+'/pfam_output')
	get_fasta_files(sequence_list, input_folder)
	for x in range(len(sequence_list)):
		run_hmmscan(input_folder, output_folder, x)
	funcd_list = pfam_post_processing(output_folder, len(sequence_list))
	return np.array(funcd_list)

	


# The following function is used to get the sequence encoding feature

def get_seq_encoding(sequence_list):
	'''
	this function would output the encoding list for the input sequence
	'''
	encoding_list = pfunc.protein_sequence_encoding(sequence_list,target_length=MAX_LENGTH)
	return np.array(encoding_list)




def get_esm_1b(sequence_list, feature_folder, original_ID_sequences_esm):
    '''
    this function would output the esm_1b list for the input sequence
    '''
    output_folder = check_and_mkdir(feature_folder+'/esm_1b_output')
    esm_1b_list = []
    for x in range(len(sequence_list)):
        original_ID_name = original_ID_sequences_esm[x]
        esm_i = torch.load(output_folder+'/'+str(original_ID_name)+'.pt')
        esm_i_mean = esm_i['mean_representations']        
        esm_i_mean_layer_33 = esm_i_mean[33]
        esm_i_mean_layer_33_numpy = esm_i_mean_layer_33.numpy()
        esm_1b_list.append(esm_i_mean_layer_33_numpy)
    return np.array(esm_1b_list)


