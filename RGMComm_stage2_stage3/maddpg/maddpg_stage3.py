import numpy

import numpy as np
import pandas as pd
import torch
import os
from maddpg.actor_critic_stage3 import Actor, Critic, Comm
import torch.nn.functional as F


class MADDPG:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)
        self.comm_network = [Comm(args, agent) for agent in range(self.args.n_agents)]

        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # path to load communication model parameters
        self.load_comm_path = self.args.load_comm_dir + '/' + self.args.scenario_name
        self.load_comm_path = self.load_comm_path + '/' + 'agent_%d' % agent_id

        # path to load trained critic models
        self.load_path = self.args.load_dir + '/' + self.args.scenario_name
        self.load_path = self.load_path + '/' + 'agent_%d' % agent_id

        # load critic model for critic nn
        if os.path.exists(self.load_path + '/3999_critic_params.pkl'):
            self.critic_network.load_state_dict(torch.load(self.load_path + '/3999_critic_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.load_path + '/3999_critic_params.pkl'))

        # load comm model for comm nn
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

        for other in range(self.args.n_agents):# load all agents comm for correct index
            if os.path.exists(self.args.load_comm_dir + '/tag6_inv1stage2_{}.pkl'.format(other)):
                self.comm_network[other].load_state_dict(
                    torch.load(self.args.load_comm_dir + '/tag6_inv1stage2_{}.pkl'.format(other)))
                print('Agent {} successfully loaded comm_network: {}'.format(self.agent_id,
                                                                             self.args.load_comm_dir + '/tag6_inv1stage2_{}.pkl'.format(
                                                                                 other)))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def entropy(self, p):
        if p.data.ndim == 2:
            return - torch.sum(p * torch.log(p + 1e-8)) / float(len(p.data))
        elif p.data.ndim == 1:
            return - torch.sum(p * torch.log(p + 1e-8))
        else:
            raise NotImplementedError

    def loss_equal(self, net, x):
        p_logit = net(x)
        p = F.softmax(p_logit)
        p_ave = torch.sum(p, dim=0) / x.data.shape[0]
        ent = self.entropy(p)
        return ent, -torch.sum(p_ave * torch.log(p_ave + 1e-8))

    # update the network
    def train(self, transitions, other_agents):
        for agent_id in range(self.args.n_agents):
            o = transitions['o_%d' % agent_id]
            c = transitions['c_%d' % agent_id]
            o_next = transitions['o_next_%d' % agent_id]

            o_1 = o[:, 0:8]
            o_2 = o[:, 18:22]
            o_new = np.column_stack((o_1, o_2))
            o_comm = o_new.copy()

            o_next_1 = o_next[:, 0:8]
            o_next_2 = o_next[:, 18:22]
            o_next_new = np.column_stack((o_next_1, o_next_2))
            estimate_o_next_comm = o_next_new.copy()

            for other in range(self.args.n_agents):
                if other == agent_id: continue;
                o_other_next = transitions['o_next_%d' % other]

                o_other_next_1 = o_other_next[:, 0:8]
                o_other_next_2 = o_other_next[:, 18:22]
                o_other_next_new = np.column_stack((o_other_next_1, o_other_next_2))
                tensor_o_other_next = torch.tensor(o_other_next_new, dtype=torch.float32)
                out = self.comm_network[other](tensor_o_other_next)

                comm_other = []
                for i in range(len(out)):
                    comm_label_i = torch.argmax(F.softmax(out[i]))  # apply softmax and argmax to the out of comm nn, get comm discrete label
                    pred_comm_i = comm_label_i.data.tolist()
                    comm_other.append(pred_comm_i)
                comm_other = np.array(comm_other).reshape((len(comm_other), 1))
                estimate_o_next_comm = np.column_stack((estimate_o_next_comm, comm_other))

            o_comm = np.column_stack((o_comm, c))

            transitions['o_next_comm_%d' % agent_id] = estimate_o_next_comm
            transitions['o_comm_%d' % agent_id] = o_comm

        u_corr = transitions['u_0'].T
        c_corr = transitions['c_0'].T

        for i in range(5):
            corr = np.corrcoef(u_corr[i], c_corr)
            # print(0)

        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next, o_next_comm, o_comm = [], [], [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
            o_next_comm.append(transitions['o_next_comm_%d' % agent_id])
            o_comm.append(transitions['o_comm_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next_comm[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next_comm[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o_comm[self.agent_id])
        loss_eq00, loss_eq01 = self.loss_equal(self.actor_network, o_comm[0])
        loss_eq10, loss_eq11 = self.loss_equal(self.actor_network, o_comm[1])
        loss_eq0 = loss_eq00 - 4 * loss_eq01
        loss_eq1 = loss_eq10 - 4 * loss_eq11
        loss_eq = loss_eq0 + loss_eq1

        actor_loss = - self.critic_network(o, u).mean() + 1 * loss_eq
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(), model_path + '/' + num + '_critic_params.pkl')
