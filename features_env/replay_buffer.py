import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

prioritized = True

if prioritized:
    class ReplayBuffer:
        """Fixed-size buffer to store experience tuples."""

        def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
            """Initialize a ReplayBuffer object.

            Params
            ======
                action_size (int): dimension of each action
                buffer_size (int): maximum size of buffer
                batch_size (int): size of each training batch
                seed (int): random seed
            """
            self.action_size = action_size
            self.memory = SumTree(buffer_size)
            self.batch_size = batch_size
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
            # Set alpha parameter to control the degree of uniform random 
            # sampling:
            self.alpha = alpha
            self.max_priority = 0
            self.seed = random.seed(seed)
        
        def add(self, state, action, reward, next_state, done, priority=10):
            """Add a new experience to memory including its priority to be later sampled from the experience pool"""
            
            # If the experience pool is smaller than the batch size, then assign
            # the maximum priority bu default:
            if len(self.memory) > self.batch_size + 5:
                e = self.experience(state, action, reward, next_state, done, self.max_priority)
            # Otherwise assign priority to the power of alpha:
            else:
                e = self.experience(state, action, reward, next_state, done, int(priority) ** self.alpha)
            self.memory.add(e)
        
        def update_priority(self, new_priorities, indices):
            """Updates priority of experience after learning."""
            for new_priority, index in zip(new_priorities, indices):
                old_e = self.memory[index]
                new_p = new_priority.item() ** self.alpha
                new_e = self.experience(old_e.state, old_e.action, old_e.reward, old_e.next_state, old_e.done, new_p)
                #self.memory.update(index, new_e)
                self.memory.update(index.item(), new_e)
                if new_p > self.max_priority:
                    self.max_priority = new_p
        
        def sample(self):
            """
            Sample a batch of experiences from the memory pool based on TD Error
            priority. Return indices of sampled experiences in order to update 
            their priorities after learning from them.
            """
            experiences = []
            indices = []
            sub_array_size = self.memory.get_sum() / self.batch_size
            for i in range(self.batch_size):
                choice = np.random.uniform(sub_array_size * i, sub_array_size * (i + 1))
                e, index = self.memory.retrieve(1, choice)
                experiences.append(e)
                indices.append(index)

            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
            probabilities = torch.from_numpy(np.vstack([e.priority / self.memory.get_sum() for e in experiences])).float().to(device)
            indices = torch.from_numpy(np.vstack([i for i in indices])).int().to(device)
    
            return states, actions, rewards, next_states, dones, probabilities, indices

        def __len__(self):
            """Return the current size of internal memory."""
            return len(self.memory)
    
    class SumTree:
        """
        Leaf nodes hold experiences and intermediate nodes store experience priority sums.
        Adapted from: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
        """
        
        def __init__(self, maxlen):
            """Initialize a SumTree object.

            Params
            ======
                maxlen (int): maximum size of replay buffer
            """
            self.sumList = np.zeros(maxlen*2)
            self.experiences = np.zeros(maxlen*2, dtype=object)
            self.maxlen = maxlen
            self.currentSize = 0
            # Set insertion marker for next item as first leaf
            self.tail = ((len(self.sumList)-1) // 2) + 1

        def add(self, experience):
            """Add experience to array and experience priority to sumList."""
            if self.tail == len(self.sumList):
                self.tail = ((len(self.sumList)-1) // 2) + 1
            self.experiences[self.tail] = experience
            old = self.sumList[self.tail]
            self.sumList[self.tail] = experience.priority
            if old == 0:
                change = experience.priority
                self.currentSize += 1
            else:
                change = experience.priority - old
            self.propagate(self.tail, change)
            self.tail += 1

        def propagate(self, index, change):
            """Updates sum tree to reflect change in priority of leaf."""
            parent = index // 2
            if parent == 0:
                return
            self.sumList[parent] += change
            self.propagate(parent, change)

        def get_sum(self):
            """Return total sum of priorities."""
            return self.sumList[1]

        def retrieve(self, start_index, num):
            """Return experience at index in which walking the array and summing the probabilities equals num."""
            # Return experience if we reach leaf node
            if self.left(start_index) > len(self.sumList) - 1:
                return self.experiences[start_index], start_index
            # If left sum is greater than num, we look in left subtree
            if self.sumList[self.left(start_index)] >= num:
                return self.retrieve(self.left(start_index), num)
            # If left sum is not greater than num, we subtract the left sum and look in right subtree
            return self.retrieve(self.right(start_index), num - self.sumList[self.left(start_index)])

        def update(self, index, experience):
            """Updates experience with new priority."""
            self.experiences[index] = experience
            old_e_priority = self.sumList[index]
            self.sumList[index] = experience.priority
            change = experience.priority - old_e_priority
            self.propagate(index, change)

        def left(self, index):
            return index * 2

        def right(self, index):
            return index * 2 + 1

        def __getitem__(self, index):
            return self.experiences[index]

        def __len__(self):
            return self.currentSize

else:
    class ReplayBuffer:
        """Fixed-size buffer to store experience tuples."""

        def __init__(self, action_size, buffer_size, batch_size, seed):
            """Initialize a ReplayBuffer object.

            Params
            ======
                action_size (int): dimension of each action
                buffer_size (int): maximum size of buffer
                batch_size (int): size of each training batch
                seed (int): random seed
            """
            self.action_size = action_size
            self.memory = deque(maxlen=buffer_size)  
            self.batch_size = batch_size
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
            self.seed = random.seed(seed)
        
        def add(self, state, action, reward, next_state, done):
            """Add a new experience to memory."""
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        
        def sample(self):
            """Randomly sample a batch of experiences from memory."""
            experiences = random.sample(self.memory, k=self.batch_size)

            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
    
            return (states, actions, rewards, next_states, dones)

        def __len__(self):
            """Return the current size of internal memory."""
            return len(self.memory)