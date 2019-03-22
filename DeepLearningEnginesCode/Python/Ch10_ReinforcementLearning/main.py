#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cart pole using Q-learning.
Author: Anirudh Topiwala with small modifications by JV Stone.
https://github.com/anirudhtopiwala/CartPole-Problem-Reinforcement-Learning/blob/master/Qcartpole.py
Learns to balance pole for 500 time steps in about 3000 episodes, but it is sensitive to initial conditions (ie rand seed).
"""

#S ubmission of Final Project Report : Anirudh Topiwala
# Implementing Q learning On cart Pole Problem. 
# -*- coding: utf-8 -*-
import gym
import numpy as np
import random
import math
from time import sleep

## Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v1')
#env = gym.make('CartPole-v0') # JVS v0 no good here.

# see https://github.com/openai/gym/wiki/Leaderboard#cartpole-v0
## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta') JVS

# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)

# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-1, 1] # JVS was [-2.4, 2.4]

max_pole_angle_degrees = 5 # was 15
STATE_BOUNDS[3] = [-math.radians(max_pole_angle_degrees), math.radians(max_pole_angle_degrees)]

# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

## Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

## Defining the simulation related constants
NUM_EPISODES = 5000

SOLVED_T = 1000 # number of steps to succeed. Jvs was 150

MAX_T = 1000 # max number of steps allowed, must be > SOLVED_T

STREAK_TO_END = 50 # if number of times MAX_T is achieved = STREAK_TO_END then stop.

DEBUG_MODE = False #True

def learncartpole():
    
    global episode, explore_rate
    
    random.seed(1)
    env.seed(9)
    
    # Set learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0 # num_streaks = number of consecutive times problem solved

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)

        plot_interval = 200
        
        for t in range(MAX_T):
            if episode % plot_interval == 0:
                env.render()

            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if (DEBUG_MODE):
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            if done:
               #sleep(0.5)
               #print(q_table)
               if (t >= SOLVED_T):
                   num_streaks += 1
               else:
                   num_streaks = 0
               break

            #sleep(0.25)
        print("Episode %d finished after %d time steps, learn rate %.3f" % (episode, t, learning_rate))
            
        # It's considered done when it's solved over 100 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
        #explore_rate = 0.1 # JVS
###################### End def simulate() ##############
        
def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action

def get_explore_rate(episode):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((episode+1)/25.0)))    #using Logrithmic decaying explore rate

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.4, 1.0 - math.log10((t+1)/100.0)))  #using Logrithmic decaying learning rate

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":
    learncartpole()
