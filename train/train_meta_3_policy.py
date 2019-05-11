#!Users/dhawgupta/anaconda3/bin/ python
'''
@author : Dhawal Gupta
Purpose : Flat Reinforcement Learning
Created : 28/2/19
This training regime is setup to train a flat RL policy eliminating the hierarchaigy where the state space will be including the intent space and the confidence space, also the intent space will be able include multiple intents at a time.

Here action 19 will take part as the option 5 of the meta policy to get the next pair of intents8
'''

import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np
from src.DQN.DQN1 import DQNAgent
from collections import namedtuple
from src.util import impdicts
from src.envs.flatenv import FlatEnv
from src.util import utils
from time import sleep
from datetime import datetime
from src.util.utils import bcolors
from typing import List, Tuple, Dict
import argparse

ap = argparse.ArgumentParser()
# add the training parameters
ap.add_argument('-e','--episodes',required = False ,help='The number of episodes that the  Policy should train for.) If not specified then a default method will be used to save the file', type = int,default = 1_00_000) # default is 1 lakh
ap.add_argument('-eps','--epsilon',required = False ,help='The starting epslion value for training of the Policy', type = float ,default = 1) # default is 1 lakh
ap.add_argument('-gpu','--set-gpu', required=False,help ='Give the GPU number, if GPU is to be used, otherwise dont mention for run on CPU', type = int)
ap.add_argument('-bs','--batch-size',required = False, help = "THe Batch size of the required trianing of the  policy", type= int, default = 64)
ap.add_argument('-lr','--learning-rate',required=False, default=0.05,type = float, help= 'The learning rate for  policy training')
ap.add_argument('-df','--discount-factor',required = False, default = 0.7, type = float , help = 'THe Discount factor for the learning of  polciy')
ap.add_argument('-do','--dropout',required = False,default = 0.00, type = float, help = 'The dropout probabliltiy for the  policy training')
ap.add_argument('-ns','--number-slots',required=False , help = 'The number of confidence slots in the domain', type = int, default = 8)
ap.add_argument('-ni','--number-intents',required = False, help = 'The Number of intents that the domain has', type = int, default = 5)
ap.add_argument('-no','--number-options',required=False, help='The number of options for the  policy', type = int, default = 6)
ap.add_argument('-as','--action-size', required = False, help = 'The action size of the Flat RL policy', type = int, default = 20 )
ap.add_argument('-p','--save-folder',required = True, help = 'Provide the path to the fodler where we need to store the policy',type = str, default = './save/')
ap.add_argument('-nf','--note-file', required = False, help = 'Give a special note that might be needed to add at the end of the file name id the user is not providing the dfile name', default= '')
ap.add_argument('-cf','--config-folder',required= True, help = 'Give the folder were we should save the config file ', default = './config/')
# TODO need to enable the facility of the loadname and SaveIn fcaility in the DQN1 program
ap.add_argument('-lf','--create-log',required = False, help = "Specify if requried to create a log file of the running experiments[yet to be implemented]", default = False)
ap.add_argument('-hl','--hidden-layers',required = False, help = "Mention the configuration of the hidden layers to be used on training the policy (Format : n1_n2_n3 : n1 n2 n3 are the number of nodes in the respective hidden layers", type = str, default = '75')


args  = vars(ap.parse_args())

NO_SLOTS = args['number_slots']
NO_INTENTS = args['number_intents']
STATES_SIZE = NO_SLOTS + NO_INTENTS
ACTION_SIZE = args['action_size']

def main():

    epsilon = args['epsilon']
    env = FlatEnv()
    EPISODES = args['episodes']
    a = str(datetime.now()).split('.')[0]
    hidden_layers = [int(i) for i in args['hidden_layers'].split('_')]
    Agent = DQNAgent(state_size=STATES_SIZE ,action_size= ACTION_SIZE, hiddenLayers=hidden_layers, dropout = args['dropout'], activation = 'relu',loadname = None, saveIn = False, learningRate=args['learning_rate'], discountFactor= args['discount_factor'])
    filename = args['save_folder']

    filename = "{}{}_Flat_HiddenLayers_{}_Dropout_{}_LearningRate_{}_Gamma_{}_Activation_{}_Episode_{}_Flat_rl_policy_{}.h5".format(filename, a , str(Agent.hiddenLayers), str(Agent.dropout) , str(Agent.learning_rate), str(Agent.gamma), Agent.activation, str(EPISODES), args['note_file'])

    batch_size = args['batch_size']
    track = []
    i = 0
    config_file = '{}{}.txt'.format(args['config_folder'], a) # this is the configualtion older containt all the details of the experimernt along with the files names

    with open(config_file, 'w') as fil:
        fil.write(str(args))
        fil.write('\n')
        fil.write("Flat Policy File : {}".format(filename))


    for episode in range(EPISODES):  # Episode
        running_reward = 0
        [confidence_state, intent_state] = env.reset() #
        done = False #
        while not done:  #
            #

            state = np.concatenate([confidence_state, intent_state])

            state = state.reshape([1, STATES_SIZE])  # Converted to appropritate size
            bcolors.printblue("The State : {}".format(state))
            intent_set_completed = False  # over her the option will mean the consolidated intent sapce of the iteration
            i_ = 0
            while not intent_set_completed :
                all_actions = env.constrain_actions()
                action = Agent.act(state, all_act = all_actions, epsilon = epsilon ) # provide episone for greedy approach
                confidence_state, intent_state, reward, intent_set_completed ,done= env.step( action) # the step will be  nromal step
                next_state  = np.concatenate([confidence_state, intent_state])
                next_state = np.reshape(next_state, [1, STATES_SIZE])
                epsilon = Agent.observe((next_state, action, reward, next_state, intent_set_completed), epsilon=epsilon)
                if Agent.memory.tree.total() > batch_size:
                    Agent.replay()
                Agent.rem_rew(reward)
                running_reward += reward
                if i % 100 == 0:
                    avr_rew = Agent.avg_rew()
                    track.append([str(i) + " " + str(avr_rew) + " " + str(episode) + " " + str(epsilon)])
                    with open("results_" + a + "_.txt", 'w') as fi:
                        for j in range(0, len(track)):
                            line = track[j]
                            fi.write(str(line).strip("[]''") + "\n")
                # print(track)
                if intent_set_completed:
                    print("Moving to the next set of itnent : {}".format(intent_state))
                    break
                state = next_state
             ##############################################
            if done:
                bcolors.printgreen("episode: {}/{}, Reward: {}, e's: {}".format(episode, EPISODES, running_reward, epsilon,))

                print("The state is : ", state)
                break

        if episode % 200 == 0:
            print("Episodes : {}".format(episode))
            # Saving the progress
            print("Saving")
            # convert this to save model for each policy
            Agent.save(filename)
            print("Done Saving You can Now Quit")
            sleep(0.5)




if __name__ == "__main__":
    if args['set_gpu'] is not None:
        print("Using the GPU ")
        DQNAgent.setup_gpu(int(args['set_gpu']))
    else:
        print("{LOG} Using the CPU for Computation")
        pass

    main()

