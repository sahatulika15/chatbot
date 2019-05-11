"""
This code is responsiible for training the multi intent model, where the meta policy has to serve multiple tinent, in a given user interaction with the system.

"""

import numpy as np
from collections import namedtuple
from ..DQN.hDQN_multi import *
from ..util import impdicts_multi
from ..envs.environments import MetaEnvMulti
from ..util import utils
from datetime import datetime
from time import sleep
import sys, os
sys.path.insert(0, os.path.abspath('..'))

def main():
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    a = str(datetime.now()).split('.')[0] + "_Multi_intent"
    fileMeta = "./saveMeta/{}.h5".format(a)
    fileController ="./saveController/{}.h5".format(a) # Later we can the add info of layers and nubmer of episodes
    env = MetaEnvMulti()
    agent = hDQN(saveInController=False, saveInMeta=False)
    anneal_factor = (1.0-0.1)/12000
    anneal_start_meta = 6

    for episode_thousand in range(12):

        for episode in range(1000): # Loop for each epsidoe
            total_external_reward = 0
            print("\n\n### EPISODE "  + str(episode_thousand*1000 + episode) + "###")
            [confidence_state, intent_state] = env.reset()
            done = False

            while not done: # The Loop In wihch meta policy acts
                temp1 = np.reshape(confidence_state, [-1])
                temp2 = np.reshape(intent_state, [-1])
                meta_state = np.concatenate([temp1,temp2])
                goal = agent.select_goal(meta_state) # now the goal has 6 possible actions
                print("Meta State: {} , Options Selected : {}".format(meta_state, goal))
                goal_reached = False

                env.meta_step_start(goal)

                print("##Entering Controller for {} ## ".format(impdicts_multi.indx2intent[goal]))
                goal_iter = 0
                if goal == 5: # the asking the user_agent option
                    pass
                else: # the normal process

                    while not goal_reached:
                        action = agent.select_move(confidence_state, utils.one_hot(goal, META_OPTION_SIZE), goal)
                        if action == 19:
                            print("##ENDING ACTION PICKED")
                        print("Epsiode : {}".format(episode + episode_thousand*1000))
                        print("Goal : {}, State : {}, Action : {}".format(goal, confidence_state, action))
                        next_confidence_state, intrinsic_reward, goal_reached  = env.controller_step(goal,action) # get the reward at the sub policy level
                        exp = ActorExperience(confidence_state,  utils.one_hot(goal, META_OPTION_SIZE), action, intrinsic_reward, next_confidence_state, done)
                        print(exp)
                        agent.store(exp, meta=False)
                        agent.update(meta=False)
                        agent.update(meta=True)
                        confidence_state = next_confidence_state
                        print("Goal Iteration : {}".format(goal_iter))
                        goal_iter += 1


                start_confidence_state, end_confidence_state, next_intent , external_reward, done = env.meta_step_end(goal)

                start_state_temp = np.concatenate([np.reshape(start_confidence_state,[-1]), np.reshape(intent_state,[-1])])
                end_state_temp = np.concatenate([np.reshape(end_confidence_state,[-1]), np.reshape(next_intent,[-1])])

                exp = MetaExperience(start_state_temp,  goal, external_reward, end_state_temp, done)

                intent_state = next_intent

                agent.store(exp, meta=True)
                total_external_reward += external_reward

                #Annealing
                if episode_thousand > anneal_start_meta:
                    agent.meta_epsilon -= anneal_factor
                agent.actor_epsilon[goal] -= anneal_factor
                if(agent.actor_epsilon[goal] < 0.1):
                    agent.actor_epsilon[goal] = 0.1
                print("meta_epsilon: " + str(agent.meta_epsilon))
                print("actor_epsilon {}".format(agent.actor_epsilon))


            if (episode % 100 == 99):
                print("Episodes : {}".format(episode + episode_thousand*1000))
                # Saving the progress
                print("Saving")
                agent.saveMeta(fileMeta)
                agent.saveController(fileController)
                sleep(0.2)
                print("Done Saving You can Now Quit")
                sleep(1)


if __name__ == "__main__":
    main()
