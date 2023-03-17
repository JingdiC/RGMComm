## RGMComm

+ This is a pytorch implementation developed for Return-Gap-Minimization Communication(RGMComm) algorithm 
+ It contains two folder: RGMComm_stage1 and RGMComm_stage2_stage3, see more detailed readme files under each folder:
  + STAGE 1 to collecting action-value(Q values) vectors samples from trained centralized critic;
  + STAGE 2 and STAGE 3 to train message generation module and train agents with communication message labels enabled; 
+ Evaluation environment is [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper is [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).

## Requirements

- python=3.6.5
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.1.0
