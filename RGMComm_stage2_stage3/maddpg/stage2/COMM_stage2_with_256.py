import statistics
import matplotlib.pyplot as plt

import numpy as np
from math import sqrt

import os

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init

import torch.nn.functional as F
from scipy import spatial

#arguments for comm labels training
num_obs_comm_nn = 12
num_comm_labels = 4
comm_nn_lr = 0.002
traning_steps = 70

# import our s1, Q2 data here
agent0_qtable = pd.DataFrame(pd.read_csv("./model/simple_tag_6_stage1_ex1/simple_tag_6/QTableStage1/tag_6_1000_2001_0.csv"))
agent1_qtable = pd.DataFrame(pd.read_csv("./model/simple_tag_6_stage1_ex1/simple_tag_6/QTableStage1/tag_6_1000_2001_1.csv"))
agent2_qtable = pd.DataFrame(pd.read_csv("./model/simple_tag_6_stage1_ex1/simple_tag_6/QTableStage1/tag_6_1000_2001_2.csv"))
agent3_qtable = pd.DataFrame(pd.read_csv("./model/simple_tag_6_stage1_ex1/simple_tag_6/QTableStage1/tag_6_1000_2001_3.csv"))
agent4_qtable = pd.DataFrame(pd.read_csv("./model/simple_tag_6_stage1_ex1/simple_tag_6/QTableStage1/tag_6_1000_2001_4.csv"))
agent5_qtable = pd.DataFrame(pd.read_csv("./model/simple_tag_6_stage1_ex1/simple_tag_6/QTableStage1/tag_6_1000_2001_5.csv"))

def convertQtable(q_table):
    list_o = []
    list_q = []
    for i in range(len(q_table)):
        string_o = q_table.iloc[i][1][1:-1:].split()
        string_o_1 = string_o[0:8]
        string_o_2 = string_o[18:22]
        string_o_1.extend(string_o_2)
        string_o = string_o_1
        list_o.append([float(num) for num in string_o])

        string_q = q_table.iloc[i][2][2:-2:].split('], [')
        list_q.append([float(num) for num in string_q])

    array = np.array(list_o)
    x = torch.tensor(array, dtype=torch.float32)

    return x, list_q, list_o

# define the comm network architecture
class Comm(nn.Module):
    def __init__(self, n_feature, n_class):
        super(Comm, self).__init__()
        self.fc1 = nn.Linear(n_feature, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.comm_out = nn.Linear(1200, n_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        comm_logit = self.comm_out(x)

        return comm_logit

def entropy(p):
    if p.data.ndim == 2:
        return - torch.sum(p * torch.log(p + 1e-8)) / float(len(p.data))
    elif p.data.ndim == 1:
        return - torch.sum(p * torch.log(p + 1e-8))
    else:
        raise NotImplementedError
def loss_equal(net, x):
    p_logit = net(x)
    p = F.softmax(p_logit)
    p_ave = torch.sum(p, dim=0) / x.data.shape[0]
    ent = entropy(p)
    return ent, -torch.sum(p_ave * torch.log(p_ave + 1e-8))

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):  # get 16 neighbors
    distances = list()
    for i, train_row in enumerate(train):
        dist = spatial.distance.cosine(test_row, train_row) #TODO cosine similarity
        distances.append((train_row, dist, i))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append((distances[i][0], distances[i][2]))
    return neighbors

def get_norm_by_mean(list_q):
    norm_list_q = []
    for i in range(len(list_q)):
        mean_q_i = sum(list_q[i]) / len(list_q[i])
        norm_q_i = [(number - mean_q_i) / 1 for number in list_q[i]]
        norm_list_q.append(norm_q_i)
    return norm_list_q

def get_norm_and_tanh(list_q):
    get_norm_and_tanh = []
    for i in range(len(list_q)):
        max_q_i = max(list_q[i])
        min_q_i = min(list_q[i])
        b_q_i = (max_q_i + min_q_i) / 2

        norm_q_i_before_tanh = [(number - b_q_i) / max_q_i for number in list_q[i]]
        tanh_q_i = np.tanh(norm_q_i_before_tanh)
        tanh_q_i_list = tanh_q_i.tolist()
        get_norm_and_tanh.append(tanh_q_i_list)
    return get_norm_and_tanh


def loss_lp(network, x, list_q, list_o):  # x is the state of agent 1
    out = network(x)  # use output of comm net
    comm_label = torch.max(F.softmax(out), 1)[1]  # apply softmax and argmax to the out of comm nn, get comm discrete label
    pred_comm = comm_label.data.numpy().squeeze()  # from mofan pytorch intro

    # for each input vector, find its 16 nearest neighbors
    Q_i_neighbors = []
    loss = 0
    for i in range(len(list_q)):
        norm_list_q = get_norm_and_tanh(list_q)
        Q_i_neighbor = get_neighbors(norm_list_q, norm_list_q[i], 16)
        Q_i_neighbors.append(Q_i_neighbor)

        x_i = list_o[i]
        tensor_x_i = torch.tensor(x_i, dtype=torch.float32)
        out_i = network(tensor_x_i)  # use output of comm net
        out_softmax_i = F.softmax(out_i).data.tolist()

        comm_label_i = torch.argmax(F.softmax(out_i))  # apply softmax and argmax to the out of comm nn, get comm discrete label
        pred_comm_i = comm_label_i.data.tolist()
        #mean_q_i = sum(list_q[i])/len(list_q[i])
        #var_q_i = statistics.variance(list_q[i])
        #var_2_q_i = sum((i - mean_q_i) ** 2 for i in list_q[i]) / len(list_q[i])
        #std_q_i = var_q_i ** 0.5
        #norm_q_i = [(number - mean_q_i) / 1 for number in list_q[i]]
        #list_norm_q_i = torch.tensor(norm_q_i, dtype=torch.float32).tolist()

        #tanh_q = torch.tanh(torch.tensor(norm_q_i, dtype=torch.float32)).tolist()
        loss_x_i = 0
        for ne in Q_i_neighbor:
            x_j = list_o[ne[1]]
            tensor_x_j = torch.tensor(x_j, dtype=torch.float32)
            out_j = network(tensor_x_j)  # use output of comm net
            out_softmax_j = F.softmax(out_j).data.tolist()

            comm_label_j = torch.argmax(F.softmax(out_j))  # apply softmax and argmax to the out of comm nn, get comm discrete label
            pred_comm_j = comm_label_j.data.tolist() # from mofan pytorch introx
            #print(pred_comm_j)
            ne_list = ne[0]

            #mean_ne_i = sum(ne_list) / len(ne_list)
            #var_ne_i = statistics.variance(ne_list)
            #var_2_ne_i = sum((i - mean_ne_i) ** 2 for i in ne_list) / len(ne_list)
            #std_ne_i = var_ne_i ** 0.5
            #norm_ne = [(number - mean_ne_i) / 1 for number in ne_list]
            #list_norm_ne = torch.tensor(norm_ne, dtype=torch.float32).tolist()

            #ne_tanh_q = torch.tanh(torch.tensor(norm_ne, dtype=torch.float32)).tolist()
            #dis = euclidean_distance(tanh_q, ne_tanh_q) # check this distance when label is the same and different
            dis = 1 - spatial.distance.cosine(norm_list_q[i], ne_list) # check this distance when label is the same and different
            #dis = 1 - spatial.distance.cosine(list_norm_q_i, ne_list ) # check this distance when label is the same and different

            #dis_Y_logits = euclidean_distance(out_softmax_i, out_softmax_j)


            loss_x_i += dis * (pred_comm_i - pred_comm_j)**2
            #loss_x_i += dis * dis_Y_logits


        loss += loss_x_i

    return loss

def runner(policy, x, list_q, list_o):
    out = policy.network(x)
    comm_label = []
    for ou in out:
        comm_label_i = torch.argmax(F.softmax(ou))  # apply softmax and argmax to the out of comm nn, get comm discrete label
        pred_comm_i = comm_label_i.data.tolist()
        comm_label.append(pred_comm_i)

    loss_eq1, loss_eq2 = loss_equal(policy.network, x)
    loss_eq = loss_eq1 - 4 * loss_eq2 #mutual information loss as the difference between marginal entropy and conditional entropy
    loss_part1 = loss_lp(policy.network, x, list_q, list_o)
    loss_1 = 0.0001 * loss_part1
    loss_2 = 10 * loss_eq
    loss = loss_1 + loss_2

    policy.optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    policy.optimizer.step()  # apply gradients
    losses = loss.data.tolist()  # from mofan pytorch introx




    return comm_label, losses

def label_distance_and_count_A(comm_label, list_q, agent_id):
    label_dis = dict()
    norm_tanh_list_q = get_norm_and_tanh(list_q)

    for i in range(num_comm_labels):
        label_dis[i] = []

    for i in range(len(comm_label)):
        label_dis[comm_label[i]].append(norm_tanh_list_q[i])

    sum_max_A = 0
    for i in range(num_comm_labels):
        lists = label_dis[i]
        result_list = lists[0]
        for j in range(1, len(lists)):
            current_list = lists[j]
            zipped_list = zip(result_list, current_list)
            sum_list = [x + y for (x, y) in zipped_list]
            result_list = sum_list
        max_element = max(result_list)
        sum_max_A += max_element
    print("A equals to: ", sum_max_A, "for agent number:", agent_id)

    count = []
    for i in range(num_comm_labels):
        count.append(len(label_dis[i]))

    label_num = pd.DataFrame(count)
    label_num.to_csv('./tag6_inv1label_count_stage_2_{}.csv'.format(agent_id))

    with_in_cluster_dis = []
    mean_list = []
    for key in label_dis.keys():
        if len(label_dis[key]) == 0:
            mean_list.append([])
            with_in_cluster_dis.append([])
        else :
            np_q_vector = np.array(label_dis[key])
            mean = np_q_vector.mean(axis=0)
            mean_list.append(mean)
            sum = 0
            for row in np_q_vector:
                sum += spatial.distance.cosine(mean, row)
            dis = sum / len(np_q_vector)
            with_in_cluster_dis.append(dis)

    different_label_dis = []
    for i in range(num_comm_labels):
        current_label_dis = []
        current = mean_list[i]
        if len(current) == 0:
            for j in range(num_comm_labels):
                current_label_dis.append(0)
        else:
            for j in range(num_comm_labels):
                if len(mean_list[j]) != 0:
                    dis = spatial.distance.cosine(current, mean_list[j])
                    current_label_dis.append(dis)
                else:
                    current_label_dis.append(0)
        different_label_dis.append(current_label_dis)

    for i in range(len(with_in_cluster_dis)):
        if not with_in_cluster_dis[i]:
            with_in_cluster_dis[i] = 0

    df = pd.DataFrame(different_label_dis)
    df['with_in_cluster_ave_dis'] = with_in_cluster_dis

    df.to_csv('./tag6_inv1cluster_CosDis_stage_2_{}.csv'.format(agent_id))


def q_distance(list_q, agent_id):
    overall_dis = []
    length = len(list_q)
    for i in range(length):
        current_dis = []
        for j in range(length):
            if i == j:
                current_dis.append(0)
            else:
                current_dis.append(spatial.distance.cosine(list_q[i], list_q[j]))
        overall_dis.append(current_dis)
    df = pd.DataFrame(overall_dis)
    df.to_csv('./tag6_inv1Qv_CosDis_stage_2_{}.csv'.format(agent_id))

def calculate_B_C(list_q, agent_id):
    sum_B = 0
    norm_tanh_list_q = get_norm_and_tanh(list_q)


    for row in norm_tanh_list_q:
        sum_B += max(row)
    print("B equals to: ", sum_B, "for agent number:", agent_id)

    result_list = norm_tanh_list_q[0]
    for i in range(1, len(norm_tanh_list_q)):
        current_list = norm_tanh_list_q[i]
        zipped_list = zip(result_list, current_list)
        sum_list = [x + y for (x, y) in zipped_list]
        result_list = sum_list
    max_element = max(result_list)
    print("C equals to: ", max_element, "for agent number:", agent_id)

# define the comm network

class policy:
    def __init__(self):
        self.network = Comm(n_feature=num_obs_comm_nn, n_class=num_comm_labels)  # n_feature need to be changed with input state dimension
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=comm_nn_lr, betas=(0.9, 0.99))

x_0, list_q_0, list_o_0 = convertQtable(agent0_qtable)
x_1, list_q_1, list_o_1 = convertQtable(agent1_qtable)
x_2, list_q_2, list_o_2 = convertQtable(agent2_qtable)
x_3, list_q_3, list_o_3 = convertQtable(agent3_qtable)
x_4, list_q_4, list_o_4 = convertQtable(agent4_qtable)
x_5, list_q_5, list_o_5 = convertQtable(agent5_qtable)


q_distance(list_q_0, 0)
q_distance(list_q_1, 1)
q_distance(list_q_2, 2)
q_distance(list_q_3, 3)
q_distance(list_q_4, 4)
q_distance(list_q_5, 5)

calculate_B_C(list_q_0, 0)
calculate_B_C(list_q_1, 1)
calculate_B_C(list_q_2, 2)
calculate_B_C(list_q_3, 3)
calculate_B_C(list_q_4, 4)
calculate_B_C(list_q_5, 5)

net_0 = policy()
net_1 = policy()
net_2 = policy()
net_3 = policy()
net_4 = policy()
net_5 = policy()

steps = traning_steps
loss_list_0 = []
loss_list_1 = []
loss_list_2 = []
loss_list_3 = []
loss_list_4 = []
loss_list_5 = []

for t in range(steps):
    print("Current E: " , t)
    net_0.network.eval()
    net_1.network.eval()
    net_2.network.eval()
    net_3.network.eval()
    net_4.network.eval()
    net_5.network.eval()


    comm_label_0, losses_0 = runner(net_0, x_0, list_q_0, list_o_0)
    comm_label_1, losses_1 = runner(net_1, x_1, list_q_1, list_o_1)
    comm_label_2, losses_2 = runner(net_2, x_2, list_q_2, list_o_2)
    comm_label_3, losses_3 = runner(net_3, x_3, list_q_3, list_o_3)
    comm_label_4, losses_4 = runner(net_4, x_4, list_q_4, list_o_4)
    comm_label_5, losses_5 = runner(net_5, x_5, list_q_5, list_o_5)

    loss_list_0.append(losses_0)
    loss_list_1.append(losses_1)
    loss_list_2.append(losses_2)
    loss_list_3.append(losses_3)
    loss_list_4.append(losses_4)
    loss_list_5.append(losses_5)

    net_0.network.train()
    net_1.network.train()
    net_2.network.train()
    net_3.network.train()
    net_4.network.train()
    net_5.network.train()


    if t == steps - 1:
        label_distance_and_count_A(comm_label_0, list_q_0, 0)
        label_distance_and_count_A(comm_label_1, list_q_1, 1)
        label_distance_and_count_A(comm_label_2, list_q_2, 2)
        label_distance_and_count_A(comm_label_3, list_q_3, 3)
        label_distance_and_count_A(comm_label_4, list_q_4, 4)
        label_distance_and_count_A(comm_label_5, list_q_5, 5)


        agent0_qtable['comm_label'] = comm_label_0
        agent1_qtable['comm_label'] = comm_label_1
        agent2_qtable['comm_label'] = comm_label_2
        agent3_qtable['comm_label'] = comm_label_3
        agent4_qtable['comm_label'] = comm_label_4
        agent5_qtable['comm_label'] = comm_label_5

        agent0_qtable.to_csv('./tag6_inv1qlabel_stage_2_{}.csv'.format(0))
        agent1_qtable.to_csv('./tag6_inv1qlabel_stage_2_{}.csv'.format(1))
        agent2_qtable.to_csv('./tag6_inv1qlabel_stage_2_{}.csv'.format(2))
        agent3_qtable.to_csv('./tag6_inv1qlabel_stage_2_{}.csv'.format(3))
        agent4_qtable.to_csv('./tag6_inv1qlabel_stage_2_{}.csv'.format(4))
        agent5_qtable.to_csv('./tag6_inv1qlabel_stage_2_{}.csv'.format(5))



torch.save(net_0.network.state_dict(), './tag6_inv1stage2_0.pkl')
torch.save(net_1.network.state_dict(), './tag6_inv1stage2_1.pkl')
torch.save(net_2.network.state_dict(), './tag6_inv1stage2_2.pkl')
torch.save(net_3.network.state_dict(), './tag6_inv1stage2_3.pkl')
torch.save(net_4.network.state_dict(), './tag6_inv1stage2_4.pkl')
torch.save(net_5.network.state_dict(), './tag6_inv1stage2_5.pkl')


plt.figure()
plt.plot(range(len(loss_list_0)), loss_list_0, label = 'comm 0')
plt.plot(range(len(loss_list_1)), loss_list_1, label = 'comm 1')
plt.plot(range(len(loss_list_2)), loss_list_2, label = 'comm 2')
plt.plot(range(len(loss_list_3)), loss_list_3, label = 'comm 3')
plt.plot(range(len(loss_list_4)), loss_list_4, label = 'comm 4')
plt.plot(range(len(loss_list_5)), loss_list_5, label = 'comm 5')


plt.xlabel('step' )
plt.ylabel('training loss')
plt.legend(loc='lower left')
plt.savefig('./tag6_inv1stage_2.png', format='png')
plt.show()

