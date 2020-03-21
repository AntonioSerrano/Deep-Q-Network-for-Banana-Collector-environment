# Deep Q-Network for Banana Collector environment #

This repository contains an implementation of the vanilla [Deep Q-Network](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) algorithm to solve the Banana Collector environment included in Unity's [ml-agents library](https://arxiv.org/pdf/1809.02627.pdf). The goal is to teach an AI's agent from scratch to collect the yellow bananas on the floor while avoiding the blue ones. It is written in Python and based on the PyTorch library. This project is the first one included in the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Environment description ##

The environment involves one agent to solve the episodic task at hand. The goal is to teach the agent from scratch to collect the yellow bananas on the floor while avoiding the blue ones.

The action space consists of 4 actions:

    0 - walk forward
    1 - walk backward
    2 - turn left
    3 - turn right

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. For example:

```python
[1.         0.         0.         0.         0.84408134 0.
 0.         1.         0.         0.0748472  0.         1.
 0.         0.         0.25755    1.         0.         0.
 0.         0.74177343 0.         1.         0.         0.
 0.25854847 0.         0.         1.         0.         0.09355672
 0.         1.         0.         0.         0.31969345 0.
 0.        ]
```

It should be noted that the states could be represented by pixels from the video game engine but for this assignment the non-visual features were preferred.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The task is considered solved when the agent is able to collect an average reward (over 100 episodes) of at least +13.

## Installation ##

1. Clone the DRLND Repository following [these instructions](https://github.com/udacity/deep-reinforcement-learning#dependencies). You will install PyTorch, the ML-Agents toolkit, and a few more Python packages required.
2. Download the Unity Environment. You need only to select the environment that matches your operating system:
    + Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    + Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    + Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    + Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Usage ##

To train the agent:
```python
python train_agent.py
```

To watch the agent act after training:
```python
python run_agent.py
```

## List of files ##

+ train_agent.py: script to start the agent training. Just type `python train_agent.py`
+ dqn_agent.py: it describes the classes for both the agent and the experience replay handler. It also states the values for the RL hyperparameters.
+ model.py: it contains the neural network architecture.
+ run_agent.py: once the agent has been trained, we can watch him behave according to the (optimal) learned policy. Type `python run_agent.py` to see the agent in action
+ checkpoint.pth: it contains the neural network weights learned during training. It is necessary to run the script run_agent.py 
+ report.md: a brief report that describes the learning algorithm, along with the chosen hyperparameters and the neural network architecture. It also includes a chart with the evolution of the average reward (over 100 episodes) during the training process. 
