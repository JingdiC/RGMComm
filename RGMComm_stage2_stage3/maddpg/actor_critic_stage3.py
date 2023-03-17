import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
# separate for agent 1 and agent 2;
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id] - 10 + args.comm_label_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        #d = torch.argmax(F.softmax(actions)).numpy()
        #actions[0][:] = 0.0
        #actions[0][d] = 1.0

        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value

class Comm(nn.Module):
    def __init__(self, args, agent_id):
        super(Comm, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape[agent_id] - 10, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.comm_out = nn.Linear(1200, args.num_comm_labels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        comm_logit = self.comm_out(x)

        return comm_logit
