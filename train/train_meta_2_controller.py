'''
author : Dhawal Gupta
Status  : INCOMPLETE
This progran is similar to the train_meta2, except for the provision of training the convtroller policy also, with a very low learning rate and saving it into another file.


#########################################
# FEATUREs :
1. intent-state-mod2
2. meta-reward-2
3. meta-state-1
4. controller-type-2
#############################################
I am alsio going to use a color coded otput for ewasir eeading of code.

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
from src.envs.metaenvmulti import MetaEnvMulti
from src.util import utils
from time import sleep
from datetime import datetime
from src.util.utils import bcolors
from typing import List, Tuple, Dict
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-cw','--controller-weights',required = True, help = "Weight for the controller policy model (i..e Mention  the file from whcih we need to load the qweights)")
# add the training parameters
ap.add_argument('-e','--episodes',required = False ,help='The number of episodes that the Meta Policy should train for.) If not specified then a default method will be used to save the file', type = int,default = 1_00_000) # default is 1 lakh
ap.add_argument('-mw','--meta-weights', required=False, help = "The Weights of the meta policy to be used for training (i.e. the file tobe saved for the same")
ap.add_argument('-tc','--train-controller',required = False, help = "Specify if the controller policy is to be trained and saved", default = False, type = bool)
ap.add_argument('-cf','--controller-folder',required = False, type = str, default = "./save/", help = "Provide the folder in which we have to save the new controller policy")
ap.add_argument('-ce','--controller-epsilon',required = False, help = "Specify the epsilon for the controller oolicy that will be used for its training", type = float, default = 0.1)
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
ap.add_argument('-p','--save-folder',required = True, help = 'Provide the path to the fodler where we need to store the meta policy',type = str, default = './save/')
ap.add_argument('-bcl','--break-controller-loop',help = 'Sometimes the Controller POlicy tends to be stuck in infiint loop, to remedy this we will restrict its runs to some value', type = int, default = 100)
# ap.add_argument('-ln','--loadname',required=False, help = "Give the file to be loaded in the meta policy if yoiu wish to continue training of teh model from a certain point", type = bool , default = )
ap.add_argument('-nf','--note-file', required = False, help = 'Give a special note that might be needed to add at the end of the file name id the user is not providing the dfile name', default= '')
ap.add_argument('-cf','--config-folder',required= True, help = 'Give the folder were we should save the config file ', default = './config/')
# TODO need to enable the facility of the loadname and SaveIn fcaility in the DQN1 program
ap.add_argument('-lf','--create-log',required = False, help = "Specify if requried to create a log file of the running experiments[yet to be implemented]", default = False)

args  = vars(ap.parse_args())

print(args)
# TODO build an argparse for the weights folder

NO_SLOTS = args['number_slots']
NO_INTENTS = args['number_intents']
META_OPTION_SIZE = args['number_options']
CONTROLLER_ACTION_SIZE = args['controller_action']
META_STATE_SIZE = NO_SLOTS + NO_INTENTS
CONTROLLER_STATE_SIZE = NO_SLOTS + NO_INTENTS



def main():
    epsilon = 1
    env = MetaEnvMulti()
    EPISODES = args['episodes']
    a = str(datetime.now()).split('.')[0]

    MetaAgent = DQNAgent(state_size=META_STATE_SIZE ,action_size= META_OPTION_SIZE, hiddenLayers=[75], dropout = args['dropout'], activation = 'relu',loadname = None, saveIn = False, learningRate=args['learning_rate'], discountFactor= args['discount_factor'])

    filename = args['save_folder']
    if 'meta-weights' not in args.keys():
        filename = "{}{}_Meta_HiddenLayers_{}_Dropout_{}_LearningRate_{}_Gamma_{}_Activation_{}_Episode_{}_single_nn_policy{}.h5".format(filename, a , str(MetaAgent.hiddenLayers), str(MetaAgent.dropout) , str(MetaAgent.learning_rate), str(MetaAgent.gamma), MetaAgent.activation, str(EPISODES), args['note_file'])
    else:
        filename = filename + args['meta_weights']

    nodes_hidden = [75] # defaulr value
    if 'controller_hidden' in args.keys():
        controller_hidden_config = args['controller_hidden']
        # extract the array from same
        nodes_hidden = [int(node) for node in controller_hidden_config.split('_') ]
    # load the agents for the controller , which is single for this case

    option_agent : DQNAgent = DQNAgent(state_size=CONTROLLER_STATE_SIZE ,action_size= CONTROLLER_ACTION_SIZE, hiddenLayers=nodes_hidden, dropout = 0.000, activation = 'relu',loadname = None, saveIn = False, learningRate=0.05, discountFactor= 0.7, epsilon = 0.0 )# Not mkaing an agent for the user based actions
    option_agent.load(args['controller_weights']) # Load the weight for al the controller policies

    controller_filename = ""  # filename to store the controller weights
    if args['train_controller'] :
        controller_filename = args['controller_folder']
        controller_filename = "{}{}_Controller_HiddenLayers_{}_Dropout_{}_LearningRate_{}_Gamma_{}_Activation_{}_Episode_{}_single_nn_policy_Modified.h5".format(
        controller_filename, a, str(option_agent.hiddenLayers), str(option_agent.dropout), str(option_agent.learning_rate),
        str(option_agent.gamma), option_agent.activation, str(EPISODES))

    visits = np.zeros([META_OPTION_SIZE]) # Store the number of Visits of each intentn tyope
    batch_size = 64
    track = []
    i = 0
    no_controller_breaks = 0
    config_file = '{}{}.txt'.format(args['config_folder'], a) # this is the configualtion older containt all the details of the experimernt along with the files names

    with open(config_file, 'w') as fil:
        fil.write(str(args))
        fil.write('\n')
        fil.write("meta_policy_file : {}".format(filename))


    for episode in range(EPISODES):  # Episode
        running_meta_reward = 0
        runnning_controller_reward = 0
        [confidence_state, intent_state] = env.reset() #
        done = False # Running the meta polciy
        while not done:  # Meta Policy Epsiode Loop
            # print("Round Meta : {}".format(episode)) # Probably not requried

            all_options = env.constrain_options()
            state = np.concatenate([confidence_state, intent_state])

            state = state.reshape([1, META_STATE_SIZE])  # Converted to appropritate size
            meta_start_state = state.copy()

            option = MetaAgent.act(state, all_options, epsilon=epsilon)
            next_confidence_state = env.meta_step_start(option)  # get the reward at the sub policy level
            meta_reward = 0
            bcolors.printblue("The Meta State : {}\nThe option : {}".format(meta_start_state, option))
            if option == 5: # the user agent option:
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
                    opt_actions = range(CONTROLLER_ACTION_SIZE) # Currently it is the while possible actions size
                    action = option_agent.act(controller_state, all_act = opt_actions, epsilon = 0 ) # provide episone for greedy approach
                    next_confidence_state, _, option_completed = env.controller_step(option, action)
                    next_controller_state = np.concatenate([next_confidence_state, goal_vector])
                    next_controller_state = np.reshape(next_controller_state, [1, CONTROLLER_STATE_SIZE])
                    # we dont need to store the experience replay memory for the controller policy
                    controller_state = next_controller_state
                    i_ +=  1
                    if i_ > args['break_controller_loop']:
                        no_controller_breaks +=1
                        break

                ###############################################

            confidence_state, next_confidence_state, intent_state, meta_reward , done = env.meta_step_end2(option)


            meta_end_state = np.concatenate([next_confidence_state, intent_state])
            meta_end_state = meta_end_state.reshape([1, META_STATE_SIZE])
            epsilon = MetaAgent.observe((meta_start_state, option,meta_reward, meta_end_state ,done), epsilon= epsilon )
            print("The next meta state : {}\n The reward : {}\nEpsilon : {}".format(meta_end_state, meta_reward, epsilon))
            if MetaAgent.memory.tree.total() > batch_size:
                MetaAgent.replay()
                MetaAgent.rem_rew(meta_reward)
            i += 1
            running_meta_reward = running_meta_reward + meta_reward
            if i % 100 == 0:  # calculating different variables to be outputted after every 100 time steps
                avr_rew = MetaAgent.avg_rew()
                track.append([str(i) + " " + str(avr_rew) + " " + str(episode) + " " + str(epsilon)])
                with open("results_" + a + "_.txt", 'w') as fi:
                    for j in range(0, len(track)):
                        line = track[j]
                        fi.write(str(line).strip("[]''") + "\n")
            # print(track)
            if done:
                print("episode: {}/{}, score: {}, e's: {}\nNumber of Controller breaks : {}".format(episode, EPISODES, running_meta_reward, epsilon, no_controller_breaks))

                print("The state is : ", meta_end_state)
                break


            confidence_state = next_confidence_state

        if episode % 200 == 0:
            print("Episodes : {}".format(episode))
            # Saving the progress
            print("Saving")
            # convert this to save model for each policy
            MetaAgent.save(filename)
            # agent.saveController(fileController)
            sleep(0.2)
            print("Done Saving You can Now Quit")
            sleep(1)




if __name__ == "__main__":
    if args['set_gpu'] is not None:
        print("Using the GPU ")
        DQNAgent.setup_gpu(int(args['set_gpu']))
    else:
        print("{LOG} Using the CPU for Computation")
        pass

    main()

