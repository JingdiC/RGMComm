import numpy
import pandas as pd
from tqdm import tqdm
from agent_stage3 import Agent
from common.replay_buffer_stage3 import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.special import softmax

import random


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        correlations = []
        probs = [[[] for j in range(self.args.action_shape[i] * self.args.num_comm_labels)] for i in range(self.args.n_agents)]
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            u = []
            actions = []
            c = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action, pred_comm = agent.select_action(s, agent_id, self.noise, self.epsilon)
                    #a = action.reshape((5, 1))
                    #comm_reshape = np.array(pred_comm + 1).reshape((1,1))
                    #corr = np.corrcoef(a, comm_reshape)
                    #print(corr)
                    #correlations.append(corr)
                    u.append(action)
                    c.append(pred_comm)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                #ran = np.zeros(5)
                #ran[random.randint(1, 4)] = 1
                #ran_list = ran.tolist()
                #ran_array = np.array(ran_list)
                #actions.append(ran_array)
            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, c, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')

            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)
            np.save(self.save_path + '/returns.pkl', returns)
            #pd.DataFrame(correlations).to_csv(self.save_path + '/correlations.csv')

            # if time_step > 0 and time_step % self.args.save_rate_prob == 0:
            #     self.get_prob(probs)
            #     for agent_id in range(self.args.n_agents):
            #         u_probs_path = os.path.join(self.save_path, 'probs_agent_%d' % agent_id)
            #         if not os.path.exists(u_probs_path):
            #             os.makedirs(u_probs_path)
            #         for i, pro in enumerate(probs[agent_id]):
            #             comm_label = int(i / self.args.action_shape[agent_id])
            #             action_label = i % self.args.action_shape[agent_id]
            #             path = u_probs_path + '/' + 'pro_{0}_{1}_{2}.pkl'.format(agent_id, comm_label, action_label)
            #             np.save(path, pro)

    def get_prob(self, probs):
        transitions = self.buffer.sample(self.args.batch_size)
        for agent_id in range(self.args.n_agents):
            action_space = self.args.action_shape[agent_id]
            comm_label = self.args.num_comm_labels
            prob = probs[agent_id]
            u_for_group = transitions['u_%d' % agent_id]
            c_for_group = transitions['c_%d' % agent_id].flatten()

            group_of_u = [[[] for j in range(action_space)] for i in range(comm_label + 1)]

            for i in range(len(c_for_group)):
                u_softmax = softmax(u_for_group[i])
                u_argmax = np.argmax(u_softmax)
                current_c = c_for_group[i].astype(int)
                group_of_u[current_c][u_argmax].append(u_argmax)

            for i in range(self.args.num_comm_labels):
                current_length = 0
                for k in range(self.args.action_shape[agent_id]):
                    current_length += len(group_of_u[i][k])

                if current_length == 0:
                    for j in range(self.args.action_shape[agent_id]):
                        prob[i * action_space + j].append(0)
                else:
                    for j in range(self.args.action_shape[agent_id]):
                        prob[i * action_space + j].append(len(group_of_u[i][j]) / current_length)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action, pred_comm = agent.select_action(s, agent_id, 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
