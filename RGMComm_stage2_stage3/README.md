#RGMComm Stage 2: Message Generation Module Training & Stage 3: Training with communication message labels

+ This is a pytorch implementation developed for Return-Gap-Minimization Communication(RGMComm) algorithm STAGE 2 and STAGE 3 to train message generation module and train agents with communication message labels enabled; 
+ Evaluation environment is [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper is [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).

## Requirements

- python=3.6.5
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.1.0

## Before you start
+ For stage 2 communication model training: run maddpg/stage2/COMM_stage2_with_256.py, model parameters will be saved;
+ For Q value vector which need to be loaded when training communication model, for each scenarios, Q vector samples could be access in model/Q_Value_Samples_stage1/:
  + Predator-Prey with 2 predators: model/Q_Value_Samples_stage1/simple_tag_2
  + Predator-Prey with 3 predators: model/Q_Value_Samples_stage1/simple_tag_3
  + Predator-Prey with 6 predators: model/Q_Value_Samples_stage1/simple_tag_6
  + Cooperation Navigation with 2 agents: model/Q_Value_Samples_stage1/simple_spread_2
+ Here we provide all trained communication model used in paper:
  + Predator-Prey with 2 predators: maddpg/COMM_models/simple_tag_2:
    + 2 message labels: comm_params_2labels_agent_0.pkl and comm_params_2labels_agent_1.pkl;
    + 4 message labels: comm_params_4labels_agent_0.pkl and comm_params_4labels_agent_1.pkl;
    + 8 message labels: comm_params_8labels_agent_0.pkl and comm_params_8labels_agent_1.pkl;
    + 16 message labels: comm_params_16labels_agent_0.pkl and comm_params_16labels_agent_1.pkl;
    + 32 message labels: comm_params_32labels_agent_0.pkl and comm_params_32labels_agent_1.pkl;

  + Predator-Prey with 3 predators: maddpg/COMM_models/simple_tag_3:
    + 4 message labels: comm_params_4labels_agent_0.pkl, comm_params_4labels_agent_1.pkl, comm_params_4labels_agent_2.pkl;

  + Predator-Prey with 6 predators: maddpg/COMM_models/simple_tag_3;
    + 4 message labels: comm_params_4labels_agent_i.pkl, for i=0,...,5;

  + Cooperation Navigation with 2 predators: maddpg/COMM_models/simple_spread_2:
    + 4 message labels: comm_params_4labels_agent_0.pkl and comm_params_4labels_agent_1.pkl;
    + 16 message labels: comm_params_16labels_agent_0.pkl and comm_params_16labels_agent_1.pkl;
+ After communication models are trained, load communication parameter model ended with '.pkl' and specify its location in 'load-comm-dir' in arguments_stage3.py;

+ We still load pre-trained critic models for generating Q values to be the input of the clustering algorithm to generate message labels. 
+ The pre-trained models are saved in 'model' folder, first specify the folder name in 'load-dir' command in common/arguments_stage3.py, for each scenario, the folder is:
  + Predator-Prey with 2 predators: model/simple_tag_2_preTrain
  + Predator-Prey with 3 predators: model/simple_tag_3_preTrain
  + Predator-Prey with 6 predators: model/simple_tag_6_preTrain
  + Cooperation Navigation with 2 agents: model/simple_spread_2_preTrain

+ Specify 'save-dir' in common/arguments_stage3.py, the training results will be saved in the model/'save-dir' folder;

+ The default scenario is Predator-Prey with 6 predators. There are 7 agents in simple_tag_6.py, including 6 predators and 1 prey. Other scenarios tested in our paper are available:
   + Predator-Prey with 2 predators: simple_tag_2.py
   + Predator-Prey with 3 predators: simple_tag_3.py
   + Cooperation Navigation with 2 agents: simple_spread.py

+ Directly run the main_stage3.py, then the algorithm will start training on scenario 'simple_tag_6' for 2000000 episodes, using the trained communication model, evaluated returns will be saved under path specified in common/arguments_stage3.py 'save-dir' command;
+ For example, the training results for simple_tag_6 scenario is saved in model/tag_6_stage3_ex1 folder.
