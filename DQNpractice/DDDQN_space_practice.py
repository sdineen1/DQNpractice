#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:42:16 2019

@author: shaydineen
"""

import tensorflow as tf
import numpy as np
from helper_def import stack_frames, predict_action, update_target_graph, create_enviroment
from collections import deque
from DDDQN_practice import DDDQN
from memory import Memory

env, possible_actions = create_enviroment()

stack_size = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)


### MODEL HYPERPARAMETERS
state_size = [110,84,4]      # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels) 
action_size = env.action_space.n             # 7 possible actions
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 50         # Total episodes for training
max_steps = 50000              # Max possible steps in an episode
batch_size = 64             

# FIXED Q TARGETS HYPERPARAMETERS 
max_tau = 1000 #Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
epsilon_start = 1.0            # exploration probability at start
epsilon_end = 0.01            # minimum exploration probability 
decay_rate = 0.00005            # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
## If you have GPU change to 1million
pretrain_length = 10000   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 10000       # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

tf.reset_default_graph()

dqn = DDDQN(action_size, state_size, learning_rate, 'DQNetwork')
targetnet = DDDQN(action_size, state_size, learning_rate, 'TargetNetwork')

memory = Memory(memory_size)

for i in range(pretrain_length):
    
    if i == 0:
        
        state = env.reset()
    
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
    choice = np.random.randint(0, action_size)
    action = possible_actions[choice]
    
    next_state, reward, done, _ = env.step(action)
    
    next_state , stacked_frames = stack_frames(stacked_frames, next_state, False)
    
    if done:
        
        next_state = np.zeros(state.shape)
        
        experience = state, action, reward, next_state, done

        memory.store(experience)        
        
        state = env.reset()
        
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    else:
        
        experience = state, action, reward, next_state, done
        
        memory.store(experience)
        
        state = next_state
        
        
if training == True:
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        decay_step = 0
        
        tau = 0
        
        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)
        
        for episode in range(total_episodes):
            
            step = 0
            
            state = env.reset()
            
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            
            episode_rewards = []
            
            while step < max_steps:
                
                
                step += 1
                tau += 1
                decay_step += 1
                
                action, epsilon = predict_action(state, possible_actions, action_size, dqn, sess, epsilon_start, epsilon_end, decay_rate, decay_step)
                
                next_state, reward, done, _ = env.step(action)
                
                
                episode_rewards.append(reward)
                
                if done:
                    
                    next_state = np.zeros((110,84), dtype=np.int)
                    
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    experience = state, action, reward, next_state, done
                    
                    memory.store(experience)
                    
                    total_reward = np.sum(episode_rewards)
                    
                    step = max_steps
                
                else:
                    
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    experience = state, action, reward, next_state, done
                    
                    memory.store(experience)
                    
                    state = next_state
            print(f"Episode {episode}: reward {total_reward}")
                    
            tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
            
            state_mb = np.array([each[0][0] for each in batch], ndmin = 3)
            action_mb = np.array([each[0][1] for each in batch])
            reward_mb = np.array([each[0][2] for each in batch])
            next_state_mb = np.array([each[0][3] for each in batch], ndmin=3)
            done_mb = np.array([each[0][4] for each in batch])
            
            target_Qs_batch = []
            
            Qs_next_state = sess.run(dqn.output, feed_dict = {dqn.inputs_: next_state_mb})
            
            target_Qs_next_state = sess.run(targetnet.output, feed_dict = {targetnet.inputs_: next_state_mb})
            
            for i in range(0, batch_size):
                
                terminal = done_mb[i]
                action = np.argmax(Qs_next_state[i])
                
                if done:
                    target = reward_mb[i]
                    
                else:
                    target = reward_mb[i] + gamma * target_Qs_next_state[i][action]
                
                target_Qs_batch.append(target)
            target_mb = np.array([each for each in target_Qs_batch])
            
            _, loss, abbs_error = sess.run([dqn.optim, dqn.loss, dqn.absolute_errors], feed_dict = {dqn.inputs_:state_mb,
                                                                                                    dqn.actions_ : action_mb,
                                                                                                    dqn.ISWeights: ISWeights_mb,
                                                                                                    dqn.target_Q: target_mb})
    
            memory.batch_update(tree_idx, abbs_error)
            
            if tau > max_tau:
                
                op_holder = update_target_graph()
                sess.run(op_holder)
            