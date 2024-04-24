#!/usr/bin/env python

# The model name and the model structure may up to change, but the other part would 
# remain unchange under most circumstance.
import tflearn
import tensorflow as tf
import numpy as np
import math
import pdb
import os


def predict_second_digit(test_functional_domain_encoding,test_esm_1b,first_digit):
    if first_digit[0] == 0:
        class_0_result = []
        class_0_labels = [0]
        class_0_prob_list = []
        for i in range(len(first_digit)):
            class_0_result.append(0)
            class_0_prob_list.append([0.0])
        return class_0_result, class_0_labels, class_0_prob_list
    import time
    start=time.time()
    n_class=first_digit[0]
    level=2

    model_name = '../database/DL_models/level_'+str(level)+'_class_'+str(n_class)+'_2features.ckpt'

    DROPOUT=True
    MAX_LENGTH=1250
    DOMAIN=16306
    ESM_1B=1280
    batch_size=20000
    theta=0.0002    
    
    unique_label_dict={1: list(range(1,19))+[20, 21, 97],
    2: list(range(1,11)),
    3: list(range(1,9))+[11]+[13],
    4: list(range(1,5))+[6, 99],
    5: list(range(1,6))+[99],
    6: list(range(1,7))}
    
    unique_label = unique_label_dict[first_digit[0]]
    

    if len(np.shape(test_esm_1b))==1:
        test_functional_domain_encoding = np.reshape(test_functional_domain_encoding,(1,)+np.shape(test_functional_domain_encoding))
        test_esm_1b = np.reshape(test_esm_1b,(1,)+np.shape(test_esm_1b))




    #functions to generate variables, like weight and bias
    def weight_variable(shape):
        import math
        if len(shape)>2:
            weight_std=math.sqrt(2.0/(shape[0]*shape[1]*shape[2]))
        else:
            weight_std=0.01
        initial=tf.truncated_normal(shape,stddev=weight_std)
        return tf.Variable(initial,name='weights')
    
    def bias_variable(shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial,name='bias')
    

    def weight_variable_2d_attention(shape):
        import math
        weight_std=1.0
        initial=tf.truncated_normal(shape,stddev=weight_std)
        return tf.Variable(initial,name='weights')
    
    def bias_variable_2d_attention(shape):
        initial=tf.constant(0.0,shape=shape)
        return tf.Variable(initial,name='bias')
    

    def weight_variable_3d_attention(shape):
        import math
        weight_std=1.0
        initial=tf.truncated_normal(shape,stddev=weight_std)
        return tf.Variable(initial,name='weights')
    
    def bias_variable_3d_attention(shape):
        initial=tf.constant(0.0,shape=shape)
        return tf.Variable(initial,name='bias')
    
    def conv2d(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    def aver_pool2d(x,row,col):
        return tf.nn.avg_pool(x,ksize=[1,row,col,1],strides=[1,row,col,1],padding='SAME')
    def max_pool2d(x,row,col):
        return tf.nn.max_pool(x,ksize=[1,row,col,1],strides=[1,row,col,1],padding='SAME')
    
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))
    
    def variable_summaries(var, name):
      """Attach a lot of summaries to a Tensor."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
    
    with tf.name_scope('placeholder'):
        y_=tf.placeholder(tf.float32,shape=[None,len(unique_label)])
        domain=tf.placeholder(tf.float32,shape=[None,DOMAIN])
        esm_1b=tf.placeholder(tf.float32,shape=[None,ESM_1B])
        keep_prob=tf.placeholder(tf.float32)
    

    with tf.name_scope('fine_tune_layers'):
        with tf.name_scope('functional_domain_layers'):
            with tf.name_scope('functional_domain_2d_attention'):
                domain_bn = tflearn.batch_normalization(domain)
                domain_after_2d_attention = domain_bn
            
            with tf.name_scope('functional_domain_fc_1'):
                w_dr1_domain=weight_variable([DOMAIN,256])
                b_dr1_domain=bias_variable([256])
                h_dr1_domain=tf.nn.relu(tf.matmul(domain_after_2d_attention,w_dr1_domain)+b_dr1_domain)
                h_dr1_domain=tflearn.batch_normalization(h_dr1_domain)

            with tf.name_scope('esm_1b_layers'):
                with tf.name_scope('esm_1b_2d_attention'):
                    w_2d_attention_esm_1b = weight_variable_2d_attention([ESM_1B,ESM_1B])
                    b_2d_attention_esm_1b = bias_variable_2d_attention([ESM_1B])
                    esm_1b_bn = tflearn.batch_normalization(esm_1b)
                    layer_attention_esm_1b = tf.matmul(esm_1b_bn,w_2d_attention_esm_1b)+b_2d_attention_esm_1b
                    layer_attention_esm_1b_weights = tf.nn.sigmoid(layer_attention_esm_1b)   
                    esm_1b_after_2d_attention = tf.multiply(esm_1b_bn,layer_attention_esm_1b_weights)
                    esm_1b_after_2d_attention = tflearn.batch_normalization(esm_1b_after_2d_attention)

                with tf.name_scope('esm_1b_fc_1'):
                    w_dr1_esm_1b=weight_variable([ESM_1B,256])
                    b_dr1_esm_1b=bias_variable([256])
                    h_dr1_esm_1b=tf.nn.relu(tf.matmul(esm_1b_after_2d_attention,w_dr1_esm_1b)+b_dr1_esm_1b)
                    h_dr1_esm_1b=tflearn.batch_normalization(h_dr1_esm_1b)
            
                with tf.name_scope('esm_1b_fc_2'):
                    w_dr2_esm_1b=weight_variable([256,256])
                    b_dr2_esm_1b=bias_variable([256])
                    h_dr2_esm_1b=tf.nn.relu(tf.matmul(h_dr1_esm_1b,w_dr2_esm_1b)+b_dr2_esm_1b)
                    h_dr2_esm_1b=tflearn.batch_normalization(h_dr2_esm_1b)
    

        with tf.name_scope('densely_connected_layers'):
            with tf.name_scope('fc_1'):
                b_fc1=bias_variable([256])

                w_fc1_domain=weight_variable([256,256])
                w_fc1_esm_1b=weight_variable([256,256])

                w_shortCut_domain=weight_variable([16306,256])
                b_shortCut_domain=bias_variable([256])
                h_shortCut_domain=tf.nn.relu(tf.matmul(domain_after_2d_attention,w_shortCut_domain)+b_shortCut_domain)
                h_shortCut_domain=tflearn.batch_normalization(h_shortCut_domain)
                h_dr1_domain=tf.add(h_shortCut_domain,h_dr1_domain)  
                
                w_shortCut_esm_1b=weight_variable([1280,256])
                b_shortCut_esm_1b=bias_variable([256])
                h_shortCut_esm_1b=tf.nn.relu(tf.matmul(esm_1b_after_2d_attention,w_shortCut_esm_1b)+b_shortCut_esm_1b)
                h_shortCut_esm_1b=tflearn.batch_normalization(h_shortCut_esm_1b)
                h_dr2_esm_1b=tf.add(h_shortCut_esm_1b,h_dr2_esm_1b)    
                
                h_fc1=tf.nn.relu(tf.matmul(h_dr1_domain,w_fc1_domain)+tf.matmul(h_dr2_esm_1b,w_fc1_esm_1b)+b_fc1)
                h_fc1=tflearn.batch_normalization(h_fc1)
                if DROPOUT==True:
                    h_fc1=tf.nn.dropout(h_fc1,keep_prob)

        with tf.name_scope('softmax_layer'):
            w_fc4=weight_variable([256,len(unique_label)])
            b_fc4=bias_variable([len(unique_label)])
            y_conv_logit=tf.matmul(h_fc1,w_fc4)+b_fc4
            y_conv=tf.nn.softmax(y_conv_logit)
            ##normal softmax end


    #DEFINE EVALUATION
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            predicted_label=tf.argmax(y_conv,1)



    sess.run(tf.global_variables_initializer())
    # pdb.set_trace()
    saver = tf.train.Saver()
    saver.restore(sess, model_name)




    def whole_set_check():
        predict_test_label=[]
        prob_out_list = []
        number_of_full_batch=int(math.floor(len(test_esm_1b)/batch_size))
        for i in range(number_of_full_batch):
            predicted_label_out, prob_out = sess.run([predicted_label, y_conv],
                feed_dict={domain: test_functional_domain_encoding[i*batch_size:(i+1)*batch_size], 
                esm_1b: test_esm_1b[i*batch_size:(i+1)*batch_size], keep_prob: 1.0})
            predicted_label_out_list = list(predicted_label_out)
            predict_test_label+=predicted_label_out_list
            prob_out_current_list = list(prob_out)
            prob_out_list+=prob_out_current_list
        
        predicted_label_out, prob_out = sess.run([predicted_label, y_conv],
            feed_dict={domain: test_functional_domain_encoding[number_of_full_batch*batch_size:],
            esm_1b: test_esm_1b[number_of_full_batch*batch_size:], keep_prob: 1.0})

        # pdb.set_trace()
        predicted_label_out_list = list(predicted_label_out)
        predict_test_label+=predicted_label_out_list
        prob_out_current_list = list(prob_out)
        prob_out_list+=prob_out_current_list

        return predict_test_label, prob_out_list


    predict_label, prob_list = whole_set_check()

    # convert the predicted label to the real EC number
    predict_digit = map(lambda x: unique_label[x], predict_label)
    print("level %d train and prediction end.\n\n"%level)

    end=time.time()
    print("Running time %d min"%((end-start)/60))
    sess.close()
    return list(predict_digit), unique_label, prob_list





