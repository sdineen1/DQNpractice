#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:48:51 2019

@author: shaydineen
"""
from SumTree import SumTree
import numpy as np

class Memory(object): # stored as ( s, a, r, s_ ) in SumTree
    
    #memory hyperparameters
    PER_e = 0.01  # constant that ensures that no experience has 0 probability of being taken
    PER_a = 0.6  # Hyperparameter used to reintroduce some randomness in the experience selection for the replay buffer //if a = 0 pure uniform randomness and if a =1 only select experiences with the highest priority
    PER_b = 0.4  # controls how much the ISWeights_ affect learning// close to 0 at the begining of learning and gradually increased to 1 over the duration of training because these weights are more important at the end of learning when our q values begin to converge
    
    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error
    
    def __init__(self, capacity):
        
        self.tree = SumTree(capacity)
        
    def store(self, experience):
        
        #find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:]) #the negative '[-self.tree.capacity:' basicaslly means start a to 0 and go to the capacity// in indexing a negative number means start from the index specified in the second arguement minus the specified negative number
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
            
        self.tree.add(max_priority, experience)
        
    def sample(self, n):
        
        #create a sample array that will contain the minibatch
        memory_b = []
        
        b_idx, b_ISWeights = np.empty((n, ), dtype = np.int32), np.empty((n,1), dtype = np.float32)
        
        #calculate the priority segment
        #as explained in the paper we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n #priority segment
        
        self.PER_b = np.min([1.0, self.PER_b + self.PER_b_increment_per_sampling] ) #max = 1.0
        
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:])/self.tree.total_priority
        max_weight = (n * p_min) ** (-self.PER_b) #since its to the negative power it'll put n * p_min in the denominator
        
        for i in range(n):
            
            #A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i+1)
            value = np.random.uniform(a, b)
            
            #Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i,0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            
            b_idx[i] = index
            
            experience = [data]
            
            memory_b.append(experience)
        
        return b_idx, memory_b, b_ISWeights
    
    def batch_update(self, tree_idx, abs_errors):
        
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        
        for ti, p in zip(tree_idx, ps):
            
            self.tree.update(ti, p)
           
