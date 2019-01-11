#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:29:54 2019

@author: shaydineen
"""
import numpy as np

class SumTree(object):
    
    data_pointer = 0
    def __init__(self, capacity):
        
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        
        self.tree = np.zeros(2 * self.capacity - 1) #number of total nodes in the tree
        
        self.data = np.zeros(self.capacity, dtype = object)
        
    def update(self, tree_index, priority):
        
        #change in new priority score minus old priority score
        #tree index is the index of the old priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        while tree_index != 0:
            
            '''
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            '''
            
            tree_index = (tree_index -1) //2       
            self.tree[tree_index] += change
        
    
    def add(self, priority, data):
        
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        #we will feel leaves from left to right
        
        # Update data frame
        self.data[self.data_pointer] = data
               
        # Update the leaf
        self.update(tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity: # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
            
    def get_leaf(self, v):
        
        parent_index = 0
        
        while True:
            
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1 
            
            #if we reach the bottom end the loop
            if left_child_index >= len(self.tree):
                
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                
                else: 
                
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node
            
        
        
        
        
        
        