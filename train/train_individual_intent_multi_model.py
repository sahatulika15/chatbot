'''
This code is supposed to train indiviual models for each intent (i.e. opition)
This code is built to train a separate model for each intent. But train all intents togehter in a single manner ,i.e. by trandomly sampling from
the intent space
Removing the printing output, to fasten the execution
'''

import numpy as np
from collections import namedtuple
from ..DQN.DQN1 import DQNAgent
from ..util import impdicts
from ..envs.environments import ControllerEnv
from ..util import utils
from time import sleep
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.abspath('..'))

NO_SLOTS = 8
META_STATE_SIZE = 5
META_OPTION_SIZE = 5
CONTROLLER_STATE_SIZE = NO_SLOTS
CONTROLLER_ACTION_SIZE = 20


def main():
    filename = "./save/"
    # ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    # MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    epsilons = np.ones([META_OPTION_SIZE])
    env = ControllerEnv()  # TODO
    EPISODES = 12000
    a = str(datetime.now()).split('.')[0]
    agents = [ DQNAgent(state_size=CONTROLLER_STATE_SIZE ,action_size= CONTROLLER_ACTION_SIZE, hiddenLayers=[30,30,30], dropout = 0.000, activation = 'relu',loadname = None, saveIn = False, learningRate=0.05, discountFactor= 0.7 ) for i in range(META_OPTION_SIZE)]


    filename = "{}_{}_HiddenLayers_{}_Dropout_{}_LearningRate_{}_Gamma_{}_Activation_{}_Episode_{}".format(filename, a ,str(agents[0].hiddenLayers), str(agents[0].dropout) , str(agents[0].learning_rate), str(agents[0].gamma), agents[0].activation, str(EPISODES))
    filename_policy=  [filename + '_option_{}.h5'.format(i) for i in range(META_OPTION_SIZE)] # list of filenames for each option


    visits = np.zeros([META_OPTION_SIZE]) # Store the number of Visits of each intentn tyope
    i_goals = np.zeros([META_OPTION_SIZE])
    batch_size = 32
    track = []
    i = 0
    for episode in range(EPISODES):
        print("\n\n### EPISODE " + str(episode) + "###")
        # Get a random option
        goal = np.random.randint(META_OPTION_SIZE) # randomly sample a option to pursue
        visits[goal] +=1
        running_reward = 0
        [confidence_state, goal_vector] = env.reset(goal = goal)
        # visits[episode_thousand][state] += 1
        done = False
        while not done:  # The Loop i whcih meta policy acts
            all_acts = env.constrain_actions() # Still function will constrain the set of actions that are required
            # now we need to convert it to stat
            # state = np.concatenate([confidence_state, goal_vector])
            state = np.array(confidence_state) # not using the goal vector
            state = state.reshape([1, CONTROLLER_STATE_SIZE])  # Converted to appropritate size
            action = agents[goal].act(state, all_acts, epsilon=epsilons[goal])
            next_confidence_state, reward, done = env.step(action)  # get the reward at the sub policy level
            # next_state = np.concatenate([next_confidence_state, goal_vector])
            next_state = np.array(next_confidence_state)
            next_state = next_state.reshape([1, CONTROLLER_STATE_SIZE])
            epsilons[goal] = agents[goal].observe((state, action,reward, next_state,done), epsilon= epsilons[goal])
            if agents[goal].memory.tree.total() > batch_size:
                agents[goal].replay()
            agents[goal].rem_rew(reward)
            i_goals[goal] += 1
            running_reward = running_reward + reward
            print("Episod=" + str(episode))
            print("i=" + str(i_goals[goal]))

            # print("State : {}\nAction : {}\nNextState : {}\n".format(state,action,next_state))
            if i_goals[goal] % 100 == 0:  # calculating different variables to be outputted after every 100 time steps
                avr_rew = agents[goal].avg_rew()
                track.append([str(i) + " " + str(avr_rew) + " " + str(episode) + " " + str(epsilons[goal])])
                with open("results_" + a + "_Policy_{}_.txt".format(goal), 'w') as fi:
                    for j in range(0, len(track)):
                        line = track[j]
                        fi.write(str(line).strip("[]''") + "\n")
            # print(track)
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(episode, EPISODES, running_reward, epsilons[goal]))
                print("The state is : ", next_state)
                break

            confidence_state = next_confidence_state
        if episode % 100 == 0:
            print("Episodes : {}".format(episode))
            # Saving the progress
            print("Saving")
            # convert this to save model for each policy
            # save the model for all agents
            for i in range(META_OPTION_SIZE):
                agents[i].save(filename_policy[i])
            # agent.save(filename)
            # agent.saveController(fileController)
            sleep(0.2)
            print("Done Saving You can Now Quit")
            sleep(1)




def run_experiment():
    """
    This will be the same function as above an will be used to call multiple iterations of programs with varying parameters

    :return:
    """
    pass

if __name__ == "__main__":
    main()
