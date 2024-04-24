import os
#processing the output file to get the predicted label
def get_label_from_output(filename):
    with open(filename,'r') as file:
        text=file.read()
        lines=text.splitlines()
        try:
            nearest_line_index=lines.index('Scores for complete sequences (score includes all domains):')+4
            label=lines[nearest_line_index].split()[-1]
        except:
            label=-1
    return label


# def fourth_num_predict(test_input_dir, sequence):
def fourth_num_predict(test_input_dir, sequence, first_digit):
    
    if first_digit == 0:
        return 0
    
    with open(test_input_dir+'/temp.fasta','w') as file:
        file.write('>0\n'+sequence)

    cmd='phmmer '
    cmd+=test_input_dir+'/temp.fasta '
    cmd+='../database/pfam/hmm.fasta'+' '
    cmd+='> '+test_input_dir+'/temp.out'
    # print cmd
    os.system(cmd)

    label = get_label_from_output(test_input_dir+'/temp.out')
    if label==-1:
        return 0
    else:
        if label.split('.')[-1].isdigit():
            return int(label.split('.')[-1])
        else:
            return 0











