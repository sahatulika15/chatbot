import numpy as np
from collections import namedtuple
from ..DQN.hDQN import hDQN
from ..util import impdicts
from ..envs.environments import MetaEnv
from ..util import utils
import sys

import sys, os
sys.path.insert(0, os.path.abspath('..'))


if len(sys.argv) > 2:
    # the filename has been given
    fileMeta = sys.argv[1]  # contains the fielname
    fileController = sys.argv[2]
else:
    fileMeta = './saveMeta/2019-01-28 17:26:42.h5'
    fileController = './saveController/2019-01-28 17:26:42.h5'
print("Loading FileMeta {}".format(fileMeta))
print("Loading FileController {}".format(fileController))
'''
eps = None
if len(sys.argv) > 3:
    eps = sys.argv[3]  # the number of episodes to run for
time.sleep(1)
'''
NO_SLOTS = 8
META_STATE_SIZE = 5
META_OPTION_SIZE = 5
CONTROLLER_STATE_SIZE = NO_SLOTS + META_STATE_SIZE
CONTROLLER_ACTION_SIZE = 20

def main():
    #ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    #MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    #a = str(datetime.now()).split('.')[0]
    #fileMeta = "./saveMeta/{}.h5".format(a)
    #fileController ="./saveController/{}.h5".format(a) # Later we can the add info of layers and nubmer of episodes
    env = MetaEnv()
    agent = hDQN()
    #env.load(fileMeta)
    agent.load(fileController, meta=False)
    agent.load(fileMeta, meta=True)
    visits = np.zeros((12, 6)) # not required for me
    #anneal_factor = (1.0-0.1)/12000
    #print("Annealing factor: " + str(anneal_factor))
    #anneal_start_meta = 6 # start the annealing after 6000 epsidoes 
    for episode_thousand in range(1):
        for episode in range(1): # Loop for each epsidoe
            total_external_reward = 0
            print("\n\n### EPISODE "  + str(episode_thousand*1000 + episode) + "###")
            [confidence_state, intent_state] = env.reset() # TODO
            # visits[episode_thousand][state] += 1
            done = False
            agent.meta_epsilon = 0.05
            while not done: # The Loop i whcih meta policy acts
                goal = agent.select_goal(intent_state)
                agent.actor_epsilon[goal] = 0.05
                print("Meta State: {} , Options Selected : {}".format(intent_state, goal))
                external_reward = 0
                goal_reached = False
                goal_start_state = confidence_state # the starting state in the beginning of a goal
                # Loop in which a sub policy acts for a goal
                env.meta_step_start(goal)
                print("##Entering Controller for {} ## ".format(impdicts.indx2intent[goal]))
                goal_iter = 0
                while not goal_reached:
                    action = agent.select_move(confidence_state, utils.one_hot(goal, META_OPTION_SIZE), goal)
                    if action == 19:
                        print("##ENDING ACTION PICKED")
                    print("Epsiode : {}".format(episode + episode_thousand*1000))
                    print("Goal : {}, State : {}, Action : {}".format(goal, confidence_state, action))
                    next_confidence_state, intrinsic_reward, goal_reached  = env.controller_step(goal,action) # get the reward at the sub policy level
                    #exp = ActorExperience(confidence_state,  util.one_hot(goal, META_OPTION_SIZE), action, intrinsic_reward, next_confidence_state, done)
                    #print(exp)
                    #agent.store(exp, meta=False)
                    #agent.update(meta=False)
                    #agent.update(meta=True)
                    confidence_state = next_confidence_state
                    # if goal_iter%100 == 0:
                    print("Goal Iteration : {}".format(goal_iter))
                    goal_iter += 1
                # calculate the Meta Reward
                goal_end_state = confidence_state # the ending state in a goal
                next_intent , external_reward, done = env.meta_step_end(goal)
                # external_reward = env.calculate_external_reward(goal_start_state, goal_end_state, goal)
                #exp = MetaExperience(intent_state,  goal, external_reward, next_intent, done)
                intent_state = next_intent
                #agent.store(exp, meta=True)
                total_external_reward += external_reward
                print(done)
                print("===============================================================================")                
                #Annealing 
                #if episode_thousand > anneal_start_meta:
                #    agent.meta_epsilon -= anneal_factor
                #agent.actor_epsilon[goal] -= anneal_factor
                #if(agent.actor_epsilon[goal] < 0.1):
                #    agent.actor_epsilon[goal] = 0.1
                print("meta_epsilon: " + str(agent.meta_epsilon))
                print("actor_epsilon {}".format(agent.actor_epsilon))
if __name__ == "__main__":
    main()
