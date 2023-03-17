#RGMComm Stage 1: Collecting Q value vectors for each agent

+ This is a pytorch implementation developed for Return-Gap-Minimization Communication(RGMComm) algorithm STAGE 1 to collecting action-value(Q values) vectors samples.
+ Evaluation environment is [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper is [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).

## Requirements

- python=3.6.5
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.1.0

## Quick Start
+ First use any Centralized Training Decentralized Execution algorithm to train the target scenario until converge, this is for getting the joint state-action values from pre-trained centralized critic, here we use a pytorch implementation of [MADDPG](https://github.com/starry-sky6688/MADDPG);
+ Save the pre-trained model under 'model' folder and specify the folder name in 'load-dir' command in common/arguments.py; here we listed folders evaluated in our paper and provide trained learning curves for each scenario: 
  + Predator-Prey with 2 predators: model/simple_tag_2_preTrain
  + Predator-Prey with 3 predators: model/simple_tag_3_preTrain
  + Predator-Prey with 6 predators: model/simple_tag_6_preTrain
  + Cooperation Navigation with 2 agents: model/simple_spread_2_preTrain

+ Directly run the main.py, then the algorithm will be collecting Q value samples on target scenario using the pretrained model, collected Q value vectors will be saved under path specified in common/arguments.py 'save-dir' command;

## Note
+ The default scenario is Predator-Prey with 6 predators. There are 7 agents in simple_tag_6.py, including 6 predators and 1 prey. Other scenarios tested in our paper are available:
   + Predator-Prey with 2 predators: simple_tag_2.py
   + Predator-Prey with 3 predators: simple_tag_3.py
   + Cooperation Navigation with 2 agents: simple_spread.py
+ Any CTDE algorithm could be compatible with this algorithm, please save actor and critic parameters and change the model load path in maddpg/maddpg.py;
+ Specify 'save-dir' in common/arguments.py, the sampled Q vectors will be saved in the model/'save-dir' folder;
