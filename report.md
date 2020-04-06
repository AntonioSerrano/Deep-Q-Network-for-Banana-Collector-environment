## List of files in each environment ##

+ train_agent.py: script to start the agent training.
+ dqn_agent.py: it describes the class to manage the agent, including the values for the RL hyperparameters.
+ replay_buffer.py: it describes the class to handle the experience replay database. 
+ model.py: it contains the neural network architecture.
+ run_agent.py: once the agent has been trained, we can watch him behave according to the (optimal) learned policy.
+ checkpoint.pth: it contains the neural network weights learned during training. It is necessary to run the script run_agent.py 

## Deep Q-Network operation

Deep Q-Network (DQN) is a Reinforcement Learning (RL) algorithm that successfully combined the [Q-learning algorithm](https://link.springer.com/article/10.1007/BF00992698) with a neural networks for the first time back in 2013. A single Convolutional Neural Network (CNN) was able to learn from scratch how to play several video games from the Atari 2600 console better than professional human players in most of those games.

In RL, an agent interacts with a given environment trying to learn a policy function (i.e. a function that tells the agent how to act) to maximize the sum of some kind of cumulative reward. The agent learns across a set of subsequent episodes comprising a finite number of time-steps. At a given time-step t, he observes a state s<sub>t</sub> , takes an action a<sub>t</sub>, obtains a reward r<sub>t</sub> from the environment, and transitions to the next state s<sub>t+1</sub>.

The agent tries to learn the optimal Q-value function that tell him the quality of each possible action for a given state. In each iteration of the algorithm, these Q-values are updated following this rule:

<p align="center"> 
    <img src="./media/q_updating_rule.gif">
</p>

Additionally, a CNN is used to approximate the Q-value function. Those weights are updated iteratively during the training process. In every of those iterations, the goal is to minimize the loss function below:

<p align="center"> 
    <img src="./media/dqn_loss_function.gif">
</p>

In each timestep, experience tuples (s<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>, s<sub>t+1</sub>, bool_episode_done) are fed to the neural network to output the Q-values for every state-action pair. 

At the beginning, the actions are taken randomly. But the more the agent explores the environment and the predictions from the neural network become more reliable, the more he uses those predictions to take actions (thanks to an epsilon-greedy strategy).

Additional measures to enable and speed up convergence of the algorithm are adopted, such as the use of an experience replay database, the target network with frozen weights, and reward clipping. But they are not discussed here.

## Chosen hyperparameters

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network

## Neural network architecture

+ Banana environment: a feedforward neural network with two hidden layers plus the final layer. The hidden layers are made of 64 units each. The final layer outputs a predicted Q-value for each of the possible 4 actions. The non-linear activation function selected is ReLU. No dropout layers were added to prevent overfitting.

+ Visual Banana environment: it includes the same CNN architcture as in [Mnih et al. (2015)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) i.e. 3 convolutional layers (with no pooling or dropout layers) and 2 fully-connected layers at the end.

As for the optimizer, Adam was chosen to do the job.

## Training results

The following plot shows the evolution of the average reward (over 100 episodes) during training over 2,000 episodes for the Banana environment . As it can be seen, the agent steadily learns how to maximize the average reward. In particular, it reaches the goal of +13 around episode no. 500:

<center><img src="./media/average_reward.png"></center>

## Future Work

+ Run Visual Banana environment with no_graphic option. As far as I know, the ml-agents library force you to render the environment to use visual observations. The drawback of this is that it affects performance since you are partially spending resources for rendering. Additionally, you must configure an X server in case you want to run the visual environment from a remote machine (good luck with that...).
+ Implement the frame skipping technique for VisualBanana environment.
+ Meta-Learning: add some kind of [Meta-Learning](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(19)30061-0) scheme as in [Duan et al. (2016)](https://arxiv.org/pdf/1611.02779) based on Recurrent Neural Networks to be able to solve similar tasks avoiding training from scratch. Of course, first it would be necessary to create other Unity environments with similar tasks.
