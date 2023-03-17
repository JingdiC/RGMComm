import random

import numpy as np
import torch
import os
from maddpg.maddpg_stage3 import MADDPG
import torch.nn.functional as F


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, agent_id, noise_rate, epsilon):
        comm_n = []
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
            #u = np.zeros(5)
            #u[random.randint(0, 4)] = 1
            comm_n = [self.args.num_comm_labels for i in range(self.args.n_agents - 1)]
        else:
            current_input_1 = o[agent_id][0:8]
            current_input_2 = o[agent_id][18:22]
            current_input_new = np.concatenate((current_input_1, current_input_2))

            for other in range(self.args.n_agents):
                if other == agent_id: continue
                other_o = o[other]
                comm_input = torch.tensor(other_o, dtype=torch.float32)
                comm_input_1 = comm_input[0:8]
                comm_input_2 = comm_input[18:22]
                comm_input_new = torch.cat((comm_input_1, comm_input_2))
                out_i = self.policy.comm_network[other](comm_input_new)  # use output of comm net
                comm_label_i = torch.argmax(F.softmax(out_i))  # apply softmax and argmax to the out of comm nn, get comm discrete label
                pred_comm_i = comm_label_i.data.tolist()
                current_input_new = np.append(current_input_new, pred_comm_i)
                comm_n.append(pred_comm_i)

            inputs = torch.tensor(current_input_new, dtype=torch.float32).unsqueeze(0)

            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            #u = u.astype('float64')
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy(), np.array(comm_n)

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)
