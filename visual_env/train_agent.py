'''
Vanilla Deep Q-Network algorithm (https://www.nature.com/articles/nature14236) 
to solve the the Banana Collector environment included in Unity's ml-agents 
library (https://github.com/Unity-Technologies/ml-agents). The states are 
represented by pixels.

Author of this script: Antonio Serrano (aserrano7@ucam.edu)

Example usage:

# On Mac:
python train_agent.py --env_path VisualBanana.app

# On Linux (64 bits):
python train_agent.py --env_path VisualBanana_Linux/Banana.x86_64

'''

import argparse
from unityagents import UnityEnvironment
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from dqn_agent import Agent
from replay_buffer import ReplayBuffer


NUM_FRAMES = 4


def preprocess_image(frame):
    # Apply relative luminance (https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems)
    # to convert image to grayscale:
    gray_frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
    
    # Resize image to (84, 84):
    gray_frame = gray_frame.reshape(frame.shape[1], frame.shape[2])
    
    # Transpose image to later stack:
    gray_frame = np.moveaxis(gray_frame, 1, 0)
    
    # Output shape will be (84, 84):
    return gray_frame 

def stack_frames(frame, state_buffer):
    # If we do not already have at least 4 frames, just append the same frame four times:
    if len(state_buffer) < 1:
        for i in range(NUM_FRAMES):
            state_buffer.append(frame)
    # Otherwise append the new frame (and delete the oldest one from the deque):
    else:
        state_buffer.append(frame)
    
    # Stack frames:
    stacked_frames = np.stack([state_buffer])
    # Output shape is now (1, 4, 84, 84)

    # Reshape array:
    stacked_frames = np.squeeze(stacked_frames, axis=0)
    # Output shape is now (4, 84, 84)

    # Output shape is (4, 84, 84):
    return stacked_frames

def dqn(env, brain_name, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    print_solved_flag=False            # warn when the the average reward reaches 13.0
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        #state = env.reset()
        env_info = env.reset(train_mode=True)[brain_name]

        # get frame from emulator
        frame = env_info.visual_observations[0]

        # set deque to store frames stack to input the neural network
        state_buffer = deque(maxlen=NUM_FRAMES)

        # preprocess frame and stack it to build the state
        frame = preprocess_image(frame)
        state = stack_frames(frame, state_buffer)

        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            #next_state, reward, done, _ = env.step(action)
            env_info = env.step(action)[brain_name]
            next_frame = env_info.visual_observations[0]
            next_frame = preprocess_image(next_frame)
            next_state = stack_frames(next_frame, state_buffer)
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0 and print_solved_flag==False:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print_solved_flag=True
            #torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            #break
    
    print('\nTraining finished! Saving weights...')
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    
    return scores


def main():
    # Parse arguments:	
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')	
    parser.add_argument("--env_path", type=str, help='Path to the ml-agents environment file')
    args = parser.parse_args()
        
    # instantiate environment:
    # - Make sure you don't have the same environment already opened in jupyter
    #   notebook or with other python.
    # - Do not try disabling rendering. Visual observations will not work.
    env = UnityEnvironment(file_name=args.env_path)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    ### instantiate agent
    env_info = env.reset(train_mode=False)[brain_name]

    # get action size
    action_size = brain.vector_action_space_size

    # get input size
    # fist, get frame from emulator
    frame = env_info.visual_observations[0]

    # set deque to store frames stack to input the neural network
    state_buffer = deque(maxlen=NUM_FRAMES)

    # preprocess frame and stack it to build the state
    frame = preprocess_image(frame)
    state = stack_frames(frame, state_buffer)
    state_size = state.shape

    # finally, instantiate agent
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    ###

    # start training
    scores = dqn(env, brain_name, agent)

    # close environment
    env.close()

    # Write scores in file for later plot edition:
    #np.save("scores", scores)  

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    print("To finish the program, manually close the plot window.")
    plt.show()
    print("Done!")


if __name__ == '__main__':
    main()
