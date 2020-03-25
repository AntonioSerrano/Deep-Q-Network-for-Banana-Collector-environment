from unityagents import UnityEnvironment
import torch

from dqn_agent import Agent


def main():
    env = UnityEnvironment(file_name="./../Banana.app")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Instantiate agent:
    env_info = env.reset(train_mode=False)[brain_name]
    #agent = Agent(state_size=8, action_size=4, seed=0)
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    for i in range(3):
        #state = env.reset()
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        for j in range(200):
            action = agent.act(state)
            #state, reward, done, _ = env.step(action)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            if done:
                break
            state = next_state
                
    env.close()


if __name__ == '__main__':
    main()
