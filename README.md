[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Deep Q-Network for Banana Collector environment

This repository contains an implementation of the [Deep Q-Network(DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) algorithm to solve the Banana Collector environment included in Unity's [ml-agents library](https://arxiv.org/pdf/1809.02627.pdf). The goal is to teach an AI's agent from scratch to collect the yellow bananas on the floor while avoiding the blue ones. The implementation also includes the DQN extensions of [Double Q-learning](https://arxiv.org/pdf/1509.06461), [Dueling architecture](https://arxiv.org/pdf/1511.06581), and [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952). It is written in Python and based on the PyTorch library. This project is the first exercise included in the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Environment description

The environment involves one agent to solve the episodic task at hand. The goal is to teach the agent from scratch to collect the yellow bananas on the floor while avoiding the blue ones.

<p align="center"> 
    <img src="./media/dqn_banana_in_action.gif" height="70%" width="70%">
</p>

The action space consists of 4 actions:

    0 - walk forward
    1 - walk backward
    2 - turn left
    3 - turn right

The repository includes two separate environments according to how states are represented:
+ Banana environment: the state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. For example:

    ```python
    [1.         0.         0.         0.         0.84408134 0.
    0.         1.         0.         0.0748472  0.         1.
    0.         0.         0.25755    1.         0.         0.
    0.         0.74177343 0.         1.         0.         0.
    0.25854847 0.         0.         1.         0.         0.09355672
    0.         1.         0.         0.         0.31969345 0.
    0.        ]
    ```

+ Visual Banana environment: states are represented by the pixels from Unity's game engine. Each state is an 84 x 84 RGB image, corresponding to the agent's first-person view of the environment.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The task is considered solved when the agent is able to collect an average reward (over 100 episodes) of at least +13.

## Installation

1. Clone the DRLND Repository following [these instructions](https://github.com/udacity/deep-reinforcement-learning#dependencies). You will install PyTorch, the ML-Agents toolkit, and a few more Python packages required.
2. Download the Unity Environments. You need only to select the environment that matches your operating system:
    + Banana environment:
        + Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
        + Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
        + Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
        + Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    + Visual Banana environment:
        + Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
        + Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
        + Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
        + Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)
    
    After unzipping the file/folder, move it to the folder of the environment you want to use e.g. `features_env/Banana.app` or `visual_env/VisualBanana_Linux/`

## Usage

+ Banana environment:
    + Go to the environment folder:
        ```console
        $ cd features_env
        ```
    + If you prefer to work with Jupyter Notebook, there is a notebook called `Navigation.ipynb` for this environment based on manual features. If you prefer to work from the terminal, go to the next step.
    + To train the agent on Mac:
        ```console
        $ python train_agent.py --env_path Banana.app
        ```
    + To train the agent on Linux (64 bits):
        ```console
        $ python train_agent.py --env_path Banana_Linux/Banana.x86_64
        ```
    + To watch the agent act after training:
        ```console
        $ python run_agent.py
        ```

+ Visual Banana environment:
    + Go to the environment folder:
        ```console
        $ cd visual_env
        ```
    + To train the agent on Mac:
        ```console
        $ python train_agent.py --env_path VisualBanana.app
        ```
    + To train the agent on Linux (64 bits):
        ```console
        $ python train_agent.py --env_path VisualBanana_Linux/Banana.x86_64
        ```
    + To watch the agent act after training:
        ```console
        $ python run_agent.py
        ```

## Detailed report

[report.md](https://github.com/AntonioSerrano/Deep-Q-Network-for-Banana-Collector-environment/blob/master/report.md) describes the learning algorithm, along with the chosen hyperparameters and the neural network architecture. It also includes a chart with the evolution of the average reward (over 100 episodes) during the training process.

## Contributing

I would love pull requests from everyone that could spot bugs or add new features to this implementation. Here are some issues you can contribute in:
+ Run Visual Banana environment with no_graphic option. As far as I know, the ml-agents library force you to render the environment to use visual observations. The drawback of this is that it affects performance since you are partially spending resources for rendering. Additionally, you must configure an X server in case you want to run the visual environment from a remote machine (good luck with that...).
+ Implement the frame skipping technique for VisualBanana environment.

How to submit a pull request:

1. [Fork](https://help.github.com/articles/fork-a-repo/) this [repository](https://github.com/AntonioSerrano/Deep-Q-Network-for-Banana-Collector-environment).
2. Create a [topic branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
3. Implement your feature or bug fix.
4. Add, commit, and push your changes.
5. Submit a [pull request](https://help.github.com/articles/using-pull-requests/).

## Credits and license

This implementation of DQN has been created by [Antonio Serrano](https://github.com/AntonioSerrano) based on the [solution](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution) provided in Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). The original algorithm was developed by [Mnih and his colleagues](https://arxiv.org/pdf/1312.5602) at Google DeepMind. Likewise, [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952) implementation was borrowed from [austinsilveria](https://github.com/austinsilveria/Banana-Collection-DQN), which is based on an unsorted [sum tree model](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/) to sample the tuples from the replay buffer more efficiently.

The content of this repository is licensed under a [MIT License](https://opensource.org/licenses/MIT).
