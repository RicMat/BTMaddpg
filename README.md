# BTMaddpg

## Scenarios
The folder mpe contains the openai's multi-particle environment. 
In mpe/multiagent/scenarios there are the different scenarios used to train the agents.
We mainly concentrate on communications, so the base scenarios in which we are interested are simple_speaker_listener 
and simple_spread.

## Training
To start training, one should run train.py which is inside maddpg/experiments. 
The only necessary parameter is --scenario "scenario-name". If you add --display, it'll show the learnt policy, 
therefore, before using this flag, you need to train the agents at least once.

## Requirements
The requirements are specified in the various requirements.txt or simply run the code and see what it's asked. 
Tensorflow 1.x is required, tensorflow==1.14.0 works just fine.

## Goal
There are two agents. One, the speaker, doesn't move but tells the listener where to go. The second, the listener, 
should listen to the speaker and move to the goal.
There are three landmarks, one of which is the goal where the agent (listener) has to move as directed by the speaker.
The goal, in general, is to reach an average reward of -30, which means that the agents are correctly moving 
towards the goal position. An average reward of -50/-60 means that the agents simply ignore the messages and 
move to the middle of the three landmarks.