'''
This model is used for test example of logistic regression imbalance data
imbalance data here means when trying the data is imbalance. However in the
test the data is somewhat balanced. 


problem: what if in the test the data is also imbalance 
'''
from __furture__ import print_function
import numpy as np
import sys
import tensorflow as tf


class 2nd_order_model:
    def __init__(self,dim_input=1,dim_output=1,update_lr,meta_lr,data_type='binary'):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = update_lr
        self.meta_lr = tf.placeholder_with_default(meta_lr,())
        if data_type=='binary':
            self.dim_hidden = [40,40]
            self.loss_func = binary_entropy
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
    
    def construct_model(self,input_tensor=None,prefix='metatrain'):
        self.features = tf.placeholder(tf.float32)
        self.labels = tf.placeholder(tf.float32)

        with tf.variable_scope('model',reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()
            loss = []
            acc = []
            min_acc = []

            def task_second_order(inp,reuse=True):
                features,labels = inp
                task_accuraciesb = []
                task_output1 = self.forward(features,weights,reuse=reuse)
                task_loss1 = self.loss_func(task_output1,labels)
                
                grads = tf.gradients(task_lossa, list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(),[weights[key]-self.update_lr*gradients[key] for key in weights.keys()]))
                task_output2 = self.forward(features,fast_weights,reuse=True)
                task_loss2 = self.loss_func(task_output2,labels)
                return [task_output1,task_output2,task_loss1,task_loss2]
            
            result = tf.map_fn(task_second_order,inp=[self.features,self.labels])
            output1,output2,loss1,loss2 = result
            self.loss1 = loss1
            self.loss2 = loss2
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(loss1)
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.gvs = gvs = optimizer.compute_gradient(self.loss2)
            self.second_op = optimizer.apply(gvs)

    def binary_entropy(logits,labels):
        logits = tf.reshape(logits,[-1])
        labels = tf.reshape(labels,[-1])
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits)


    def construct_fc_weights(self):
        weights={}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input,self.dim_hidden[0]],stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))

        weights['w2'] = tf.Variable(tf.truncated_normal([self.dim_input,self.dim_hidden[1]],stddev=0.01))
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden[1]]))

        return weights

    def forward_fc(self,inp,weights,reuse=False):
        hidden = tf.matmul(inp,weights['w1'])+weights['b1']
        logits = tf.matmul(hidden,weights['w2'])+weights['b2']
        return logits

