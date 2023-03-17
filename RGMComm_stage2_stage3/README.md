## RGMComm Stage 2: Message Generation Module Training & Stage 3: Training with communication message labels

+ This is a pytorch implementation developed for Return-Gap-Minimization Communication(RGMComm) algorithm STAGE 2 and STAGE 3 to train message generation module and train agents with communication message labels enabled; 
+ Evaluation environment is [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper is [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).

## Requirements

- python=3.6.5
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.1.0

## Quick Start:
+ Stage 2: For training communication generation using online clustering algorithm:
  + First specify the path to load Q value vectors sampled from stage 1 in maddpg/stage2/COMM_stage2_with_256.py
  + Then run maddpg/stage2/COMM_stage2_with_256.py, communication model parameters will be saved;

+ Stage 3: Training target task with RGMComm algorithm:
  + load correct communication model by specifying 'load-comm-dir' in 'common/arguments_stage3.py';
  + run main_stage3.py, model will be saved to path specified by 'save-dir' in 'common/arguments_stage3.py';
  
## Note:
+ The default scenario is Predator-Prey with 6 predators. There are 7 agents in simple_tag_6.py, including 6 predators and 1 prey. Other scenarios tested in our paper are available:
   + Predator-Prey with 2 predators: simple_tag_2.py
   + Predator-Prey with 3 predators: simple_tag_3.py
   + Cooperation Navigation with 2 agents: simple_spread.py

