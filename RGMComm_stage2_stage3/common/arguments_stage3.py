import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag_6", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries") # number of good agents here for simple tag
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--lr-comm", type=float, default=1e-3, help="learning rate of comm")
    parser.add_argument("--comm-label-shape", type=int, default=5, help="the dimension of communication labels, right now only send 1 integer to other agent")
    parser.add_argument("--num-comm-labels", type=int, default=4, help="number of communication labels, how many different labels could be chosen from")


    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model/tag_6_stage3_ex1", help="directory in which training state and model should be saved")
    parser.add_argument("--load-dir", type=str, default="./model/simple_tag_6_preTrain", help="directory in which critic network model should be loaded")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--save-rate-prob", type=int, default=256, help="save probabilities between actions and comm labels once every time this many episodes are completed")

    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--load-comm-dir", type=str, default="./maddpg/stage2", help="directory in which comm model loaded")


    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=400, help="how often to evaluate model")
    args = parser.parse_args()

    return args
