'''
This code is meant to interect with teh enviornement and see if it is beahving according to the desirted behaviour.
'''
import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np
from collections import namedtuple
from src.util import impdicts
from src.envs.metaenvmulti import MetaEnvMulti
from src.util import utils
from src.util.utils import bcolors
from time import sleep
from datetime import datetime
from typing import List, Tuple, Dict
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-e','--episodes',required = False ,help='The number of episodes that the Meta Policy should train for.) If not specified then a default method will be used to save the file', type = int,default = 1_00_000) # default is 1 lakh
ap.add_argument('-mw','--meta-weights', required=False, help = "The Weights of the meta policy to be used for training (i.e. the file tobe saved for the same")
ap.add_argument('-gpu','--set-gpu', required=False,help ='Give the GPU number, if GPU is to be used, otherwise dont mention for run on CPU', type = int)
ap.add_argument('-bs','--batch-size',required = False, help = "THe Batch size of the required trianing of the meta policy", type= int, default = 64)
ap.add_argument('-lr','--learning-rate',required=False, default=0.05,type = float, help= 'The learning rate for meta policy training')
ap.add_argument('-df','--discount-factor',required = False, default = 0.7, type = float , help = 'THe Discount factor for the learning of meta polciy')
ap.add_argument('-do','--dropout',required = False,default = 0.00, type = float, help = 'The dropout probabliltiy for the meta policy training')
ap.add_argument('-ch', '--controller-hidden', required = False, help = "The Hidden Layers  configuration of the Controller Policym This portion is not yet implemeneted in the Code \nFormat : n1_n2_n3 ..  , tells n1 nodes in layer 1 , n2 nodes in layer 2 , n3 nodes in layer 3 and so on." , default = '75')
ap.add_argument('-ns','--number-slots',required=False , help = 'The number of confidence slots in the domain', type = int, default = 8)
ap.add_argument('-ni','--number-intents',required = False, help = 'The Number of intents that the domain has', type = int, default = 5)
ap.add_argument('-no','--number-options',required=False, help='The number of options for the meta policy', type = int, default = 6)
ap.add_argument('-ca','--controller-action', required = False, help = 'The Number of Controller Actions, although thte code is designed to work for only 20 action as of now', type = int, default = 20 )
ap.add_argument('-bcl','--break-controller-loop',help = 'Sometimes the Controller POlicy tends to be stuck in infiint loop, to remedy this we will restrict its runs to some value', type = int, default = 100)
ap.add_argument('-nf','--note-file', required = False, help = 'Give a special note that might be needed to add at the end of the file name id the user is not providing the dfile name', default= '')
ap.add_argument('-lf','--create-log',required = False, help = "Specify if requried to create a log file of the running experiments[yet to be implemented]", default = False)
args  = vars(ap.parse_args())



NO_SLOTS = args['number_slots']
NO_INTENTS = args['number_intents']
META_OPTION_SIZE = args['number_options']
CONTROLLER_ACTION_SIZE = args['controller_action']
META_STATE_SIZE = NO_SLOTS + NO_INTENTS
CONTROLLER_STATE_SIZE = NO_SLOTS + NO_INTENTS



def main():
    epsilon = 1
    env = MetaEnvMulti()  # TODO
    EPISODES = args['episodes']
    visits = np.zeros([META_OPTION_SIZE]) # Store the number of Visits of each intentn tyope
    track = []
    i =0
    for episode in range(EPISODES):  # Episode
        running_meta_reward = 0
        [confidence_state, intent_state] = env.reset() #
        done = False # Running the meta polciy
        while not done:  # Meta Policy Epsiode Loop
            state = np.concatenate([confidence_state, intent_state])
            state = state.reshape([1, META_STATE_SIZE])  # Converted to appropritate size
            meta_start_state = state.copy()
            bcolors.printblue("Meta Start State : {}".format(meta_start_state))
            option = int(input("Enter the option to be taken > "))
            next_confidence_state = env.meta_step_start(option)  # get the reward at the sub policy level
            meta_reward = 0
            if option == 5: # the user agent option: This is suppposed to be taken care by the meta end state function
                pass
            else:
            #############################################################
            # HERE COMES THE PART FOR CONTROLLER EXECUTION
                option_completed = False
                # make a one hot goal vector
                goal_vector = utils.one_hot(option, NO_INTENTS)
                i_ = 0
                controller_state = np.concatenate([next_confidence_state, goal_vector])
                controller_state = controller_state.reshape(1, CONTROLLER_STATE_SIZE)
                while not option_completed:
                    bcolors.printgreen("Controller State : {}".format(controller_state))
                    action = int(input('Enter the action for option > '))
                    # action = option_agent.act(controller_state, all_act = opt_actions, epsilon = 0 ) # provide episone for greedy approach
                    next_confidence_state, _, option_completed = env.controller_step(option, action)
                    next_controller_state = np.concatenate([next_confidence_state, goal_vector])
                    next_controller_state = np.reshape(next_controller_state, [1, CONTROLLER_STATE_SIZE])
                    bcolors.printgreen("The Next Controller State : {}".format(next_controller_state))
                    controller_state = next_controller_state
                    i_ +=  1
                    if i_ > args['break_controller_loop']:
                        no_controller_breaks +=1
                        break
            confidence_state, next_confidence_state, intent_state, meta_reward , done = env.meta_step_end2(option)
            meta_end_state = np.concatenate([next_confidence_state, intent_state])
            meta_end_state = meta_end_state.reshape([1, META_STATE_SIZE])
            bcolors.printblue("The next meta state : {}\n The reward : {}\nEpsilon : {}".format(meta_end_state, meta_reward, epsilon))
            i += 1
            running_meta_reward = running_meta_reward + meta_reward
            if done:
                bcolors.printwarning("episode: {}/{}, score: {}, e's: {}\nNumber of Controller breaks : {}".format(episode, EPISODES, running_meta_reward, epsilon, no_controller_breaks))

                print("The state is : ", meta_end_state)
                break
            confidence_state = next_confidence_state




if __name__ == "__main__":
    main()



