#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Required Libraries
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch as T


# In[2]:


#Checking for available GPU device
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name())


# # RL Coding
# 

# In[3]:


class DQN(nn.Module):
    def __init__(self,input_dims,n_actions,lr):
        super(DQN,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.CNN()
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        
        
    # Convolutional architecture for the model
    def CNN(self):
        self.fc1 = nn.Linear(4,10) #The First layer. This has 4 inputs namely cart position, cart velocity, pole position, pole velocity
        self.fc2 = nn.Linear(10,15) 
        self.fc3 = nn.Linear(15,2) #The final layer, with 2 action outputs, namely 0 and 1
        
    def forward(self,state):
        output = self.fc1(state)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        action = output
        
        return action
    

class Agent():
    def __init__(self,gamma,lr,epsilon,input_dims,n_actions,batch_size,max_mem_size=10000,eps_end=0.01):
        self.input_dims = input_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        #self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [0,1] #Change this to n_actions for generalised application
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        self.Q_eval = DQN(input_dims,n_actions,lr)
        
        self.state_memory = np.zeros((self.mem_size,*self.input_dims), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size,*self.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool)
    
    def store_transition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size #For only last 10,000 entries
        self.state_memory[index] = state
        self.new_state_memory[index] = state_ #Value for next state = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def choose_action(self,state):
        if np.random.random() > self.epsilon:
            state = T.tensor([state]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item() #.item is essentially converting the tensor datatype back to numpy. This is imp because our env takes this data type
        else:
            action = np.random.choice(self.action_space) #Here action_space = [0,1]
        
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        #self.iter_cntr += 1
        #self.epsilon = self.epsilon - self.eps_dec \
        #    if self.epsilon > self.eps_min else self.eps_min
        
    """def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr,self.mem_size)
        
        batch = np.random.choice(max_mem,self.batch_size,replace=False)
        
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        action_batch = self.action_memory[batch]
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        #self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \ else self.eps_min"""
        
        


# In[4]:


import numpy as np
batch_size=10
batch_index = np.arange(batch_size)
print(batch_index)


# # Running the defined Agent

# In[5]:


#rom utils import plot_learning_curve
import numpy as np

env = gym.make('CartPole-v1')
#print(env.observation_space)
env.reset()
agent = Agent(gamma=0.99,lr=0.001,epsilon=0.1,input_dims=[4],n_actions=[2],batch_size=64,max_mem_size=10000,eps_end=0.01) 
#The value of n_actions and input_dims is defined because of the environment we have chose. In generalised env, define it accordingly

scores,eps_history = [], []
n_games = 500

for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        #print("info", info)
        env.render()
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_
        
    scores.append(score)
    eps_history.append(agent.epsilon)
    
    avg_score = np.mean(scores[-100:])
    
    print('episode ', i, 'score%.2f' % score,
          'average score %.2f' % avg_score,
          'epsilon %.2f' % agent.epsilon)
    
#env.close()
    


# In[ ]:





# In[ ]:




