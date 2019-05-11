"""
This enviornment will be hardcoded approach for the proposed system
"""


import numpy as np
import random

from ..util import utils
from ..util import impdicts
from typing import List, Tuple, Dict

import sys, os

sys.path.insert(0, os.path.abspath('..'))
class MetaEnv:

    def __init__(self, w1 : float=  1, w2 : float = 8, w3 : float = 13,   intent_space_size : int = 5, slot_space_size : int  =  8, options_space : int = 5, primitive_action_space : int = 20 ):
        """
        Args
        :param intent_space_size: The number of intents in the system
        :param slot_space_size: The number of slot that need to be filled with the probability values
        :param options_space: The number of available options in the meta policy
        :param primitive_action_space:  The number of primitve actions available
        :param current_intents: Array containing the active intents for the current obj
        """
        self.threshold = 0.7
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3 # probably will not be using this
        self.intent_space_size = intent_space_size
        self.slot_space_size = slot_space_size
        self.options_space = options_space
        self.primitive_action_space = primitive_action_space
        self.current_obj_intent =[] # This shows all the intents that have to be served in this object
        self.slot_states = []
        self.current_intent_state= []
        self.current_slot_state = np.array([])
        self.current_intent_no = 0
        self.no_intents = 0 # the number of intents to be served i.e. len(self.current_obj_intent)
        self.goal_iter = [] # The number of iterations done for each subgoal completion
        self.reset()
        self.latest_start_confidence_start = [] # Store the last confidnce state before the start of the option play for the subpolicy
        # self.slot_states # this is the list of all the slot states encountered in runs
        # self.current_slot_state # this is state of the confidence values in the current context
        # self.current_intent_state # This contains the current intent that the env is focusing on (currently a one hot vector later, maybe changed to a composite of multiple intents
        # self.intent_states # collection of the timeline of the intent states encountered



    def reset(self, prob_random_start : float = 0.5): # just trying this return syntax
        """
        confidence_state, intent_state = env.reset()
        We need to init the object with a set of new intents

        :return: confidence_state : The slot state with confidence  values
        intent_state : The intent state for the current intent
        """
        self.current_obj_intent = []
        self.no_intents = np.random.randint(1,self.intent_space_size + 1) # Produces a number between 1 and 5 for the number of intents
        # get a starting number of intents from 0 to intent_space
        temp_intents = list(range(0, self.intent_space_size))
        for iter in range(self.no_intents):
            indx = np.random.randint(0, len(temp_intents))
            self.current_obj_intent.append(temp_intents[indx])
            del temp_intents[indx]
        # after these steps teh current_onbj_intent contains the intents and there order to be followred in sccheduling intents for the agent
        if random.random() < prob_random_start:
            # do a random init
            self.random_state_init()
        else:
            self.state_init()
        # now we will set the intital intent state of the system and also the buffer to store the intent states
        self.current_intent_no = 0 # Keeps track of the intent number being served
        self.intent_states = np.array([ utils.one_hot( self.current_obj_intent[self.current_intent_no], self.intent_space_size)]) # setting the starting intent
        self.current_intent_state = self.intent_states[-1]
        return [self.current_slot_state, self.current_intent_state]


    def random_state_init(self):
        self.slot_states = np.array([[random.random() for i in range(self.slot_space_size)]])
        self.current_slot_state = self.slot_states[-1]

    def state_init(self):
        self.slot_states = np.zeros((1, self.slot_space_size))  # initliase with zero for all values
        self.current_slot_state = self.slot_states[-1]


    def step(self, action : int):
        """
        Addressing the monster in the room, this is the holy grail of the whole system
        :param action: The primitive action that is supposed to be taken
        :return: Returns a tuple of [next_confidence_state, intent_state] , intrinsic_reward, goal_reached , done
        Rules and Assumptions
        1. Whenever I recieve a terminating action for a subgoal i.e. the 19 action i will move onto the next intent_state, and also make the goal_reached variable as true
        2. When I am out of processing the next intents, I will move onto making the done variable as true.
        3. Whenever an action comes that is out of the reason of the current intent, we won't modify the state, and we will penalize the aciton by coinsidering the negtative reward from a step that will be consumed in doing that step.
        4. Currently I am also giving the extrinsic reward when taking the terminating action, but we can later remove, thsi , this is done because we need to encourage the agent to somehow learn the terminating action.

        """
        print("Env Step")
        done = False
        goal_reached = False
        new_state = np.copy(self.current_slot_state) # copy the state of the current slot state
        current_intent = self.current_obj_intent[self.current_intent_no]
        print("Step : Current Intent : {}".format(current_intent))
        reward = 0
        if action == 19:
            """
            if the terminating action is picked
            if all slots are covered above the threshold then award otherwise penalize
            """

            if self.check_confidence_state(current_intent) :
                # give full reward
                reward = self.w2*self.calculate_external_reward(np.zeros(self.slot_space_size), self.current_slot_state, current_intent)
            else:
                reward = -self.w2*self.calculate_external_reward(self.current_slot_state, np.ones(self.slot_space_size), goal = current_intent)
            self.current_intent_no += 1 # if all the intents in the current object are over
            if self.current_intent_no >= self.no_intents:
                done = True
            else:
                self.current_intent_state = utils.one_hot(self.current_obj_intent[self.current_intent_no, self.intent_space_size])
            goal_reached = True
            return [self.current_slot_state, self.current_intent_state], reward, goal_reached, done

        # if not terminating action
        relevant_actions : List[int] = impdicts.intent2action[current_intent]
        if action not in relevant_actions:
            reward = -self.w1
        else:
            if action in impdicts.askActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2*random.random() + 0.55
                # pass # here the action will be same as the slot number
            elif action in impdicts.reaskActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    if new_state[slot] < 0.1:
                        pass
                    else:
                        new_state[slot] = (1 - new_state[slot])*0.85 + new_state[slot]
                # pass # Use the index of the list to
            elif action in impdicts.hybridActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2*random.random() + 0.55
                # pass # The hybrid action
            else:
                print("Wrong action picked up please see the system.\nExiting.....")
                sys.exit()
                # pass # put an error message in the same
            # calculate the reward
            reward = self.w2*self.calculate_external_reward(self.current_slot_state, new_state, current_intent) - self.w1 # get the reward for increase in confidecne and the decrrease in iteration
            # Although the calculate external reward is not required as we are already cross checking the things in the first if condition

        self.current_intent_state = np.copy(new_state)
        return [ self.current_intent_state, self.current_intent_state], reward, False, False

    def controller_step(self, goal, action):
        """

        :param goal: The goal the controller is fulfilling to appropriately reward it
        :param action: : The action that the agent is taking pertaining to that goal
        :return: next_confidence_state, reward, goal_completed
        """
        print("Env Controller Step")
        # done = False
        goal_reached = False
        new_state = np.copy(self.current_slot_state)  # copy the state of the current slot state
        current_intent = goal
        print("Step : Current Intent : {}".format(current_intent))
        reward = 0
        if action == 19:
            """
            if the terminating action is picked
            if all slots are covered above the threshold then award otherwise penalize
            """

            if self.check_confidence_state(current_intent):
                # give full reward
                reward = self.w2 * self.calculate_external_reward(np.zeros(self.slot_space_size),
                                                                  self.current_slot_state, current_intent)
            else:
                reward = -self.w2 * self.calculate_external_reward(self.current_slot_state,
                                                                   np.ones(self.slot_space_size), goal=current_intent)
            # shifting the below part in the meta step acitons
            # self.current_intent_no += 1  # if all the intents in the current object are over

            goal_reached = True
            return self.current_slot_state, reward, goal_reached,

        # if not terminating action
        relevant_actions: List[int] = impdicts.intent2action[current_intent]
        if action not in relevant_actions:
            reward = -self.w1
        else:
            if action in impdicts.askActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2 * random.random() + 0.55
                # pass # here the action will be same as the slot number
            elif action in impdicts.reaskActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    if new_state[slot] < 0.1:
                        pass
                    else:
                        new_state[slot] = (1 - new_state[slot]) * 0.85 + new_state[slot]
                # pass # Use the index of the list to
            elif action in impdicts.hybridActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2 * random.random() + 0.55
                # pass # The hybrid action
            else:
                print("Wrong action picked up please see the system.\nExiting.....")
                sys.exit()
                # pass # put an error message in the same
            # calculate the reward
            reward = self.w2 * self.calculate_external_reward(self.current_slot_state, new_state,
                                                              current_intent) - self.w1  # get the reward for increase in confidecne and the decrrease in iteration
            # Although the calculate external reward is not required as we are already cross checking the things in the first if condition

        self.current_slot_state = np.copy(new_state)
        return self.current_slot_state, reward, False

    def meta_step_start(self,option):
        """
        This will be responsible for storing the current_confidence-state, to make a comparison with the same in the future
        :param option: Take as intput the option chosed by the meta policy
        :return: return the current confidence state of the dialogue
        """
        # store the current_confidence state
        self.latest_start_confidence_start = np.copy(self.current_slot_state)
        return self.latest_start_confidence_start

    def meta_step_end(self, option) -> Tuple[int, float, bool]  :
        """

        :param option: The option chosen
        :return: return the  next intent, reward, done (still skeptical about returning the confidence state)
        Points :
        Currently the rewards a really simple, they dont penalize for filling other slots that might not be relevant. We can penalize these states in the future by subtracting the extra sltos filled
        """
        done = False
        reward = 0
        current_intent = self.current_obj_intent[self.current_intent_no]
        reward = self.w2*self.calculate_external_reward(np.copy(self.latest_start_confidence_start), self.current_slot_state, current_intent)
        self.current_intent_no += 1  # if all the intents in the current object are over
        new_intent = current_intent
        if self.current_intent_no >= self.no_intents:
            done = True
        else:
            self.current_intent_state = utils.one_hot(
                self.current_obj_intent[self.current_intent_no], self.intent_space_size)
            new_intent = self.current_obj_intent[self.current_intent_no]
        return self.current_intent_state, reward, done

    def calculate_external_reward(self, start_state : np.ndarray, goal_state : np.ndarray, goal : int) -> float :
        """
        This function is supposed to check the goal and then check the progress
        in the confidence values of the slots positions with respect to that goal
        :param start_state: The starting state of the confidence values of options
        :param goal_state: The ending state of the confidence values of the options
        :param goal: The goal currently working for
        :return: return the reward
        """
        relevants_slots = impdicts.intent2slots[goal] # this will return the slots to be measured
        # now we will calculate the differen from both
        diff_confidence: float = np.sum(goal_state[relevants_slots] - start_state[relevants_slots])
        # now we can multiply by weights
        return diff_confidence # or we can keep and differnt weight factor for the external agent

    def check_confidence_state(self, goal):
        """
        THis function will check all the slots pertaining to the goal and see if its satisfies the threshold, if it does so , it returns a True value, otherwise it returns a false value.
        :param goal: The goal that its currently checking for
        :return: True/False
        Rules & Points :
        1. I am using the threshold currently to analyze
        """
        slots_to_be_checked = impdicts.intent2slots[goal] # these are the slots
        return all(self.current_slot_state[slots_to_be_checked] > self.threshold)

# This enviornment is build to train controller level policie
class ControllerEnv:
    def __init__(self, goal = 0, w1 = 1, w2 = 8, w3 = 0, slot_space_size = 8, options_space = 5, primitive_action_space = 20, goal_space_size = 5): # The default goal
        self.goal = goal
        self.threshold = 0.7
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3  # probably will not be using this
        self.slot_space_size = slot_space_size
        self.options_space = options_space
        self.primitive_action_space = primitive_action_space
        self.slot_states = []
        self.current_goal_state = []
        self.current_slot_state = np.array([])
        self.goal_space_size = goal_space_size
        self.goal = goal
        self.goal_iter = []  # The number of iterations done for each subgoal completion
        self.reset(goal = goal)

    def reset(self, goal= 0, state = None, prob_random_start : float = 0.5):
        self.goal = goal

        if random.random() < prob_random_start:
            # do a random init
            self.random_state_init()
        else:
            self.state_init()
        # now we will set the intital intent state of the system and also the buffer to store the intent states
        self.goal = goal  # Keeps track of the intent number being served
        self.goal_state = np.array([utils.one_hot(self.goal,self.goal_space_size)])  # setting the starting intent
        self.current_goal_state = self.goal_state[-1]
        return [self.current_slot_state, self.current_goal_state]

    def random_state_init(self):
        self.slot_states = np.array([[random.random() for i in range(self.slot_space_size)]])
        self.current_slot_state = self.slot_states[-1]

    def state_init(self):
        self.slot_states = np.zeros((1, self.slot_space_size))  # initliase with zero for all values
        self.current_slot_state = self.slot_states[-1]

    def check_confidence_state(self, goal):
        """
        THis function will check all the slots pertaining to the goal and see if its satisfies the threshold, if it does so , it returns a True value, otherwise it returns a false value.
        :param goal: The goal that its currently checking for
        :return: True/False
        Rules & Points :
        1. I am using the threshold currently to analyze
        """
        slots_to_be_checked = impdicts.intent2slots[goal] # these are the slots
        return all(self.current_slot_state[slots_to_be_checked] > self.threshold)

    def calculate_external_reward(self, start_state : np.ndarray, goal_state : np.ndarray, goal : int) -> float :
        """
        This function is supposed to check the goal and then check the progress
        in the confidence values of the slots positions with respect to that goal
        :param start_state: The starting state of the confidence values of options
        :param goal_state: The ending state of the confidence values of the options
        :param goal: The goal currently working for
        :return: return the reward
        """
        goal = self.goal
        relevants_slots = impdicts.intent2slots[goal] # this will return the slots to be measured
        # now we will calculate the differen from both
        diff_confidence: float = np.sum(goal_state[relevants_slots] - start_state[relevants_slots])
        # now we can multiply by weights
        return diff_confidence # or we can keep and differnt weight factor for the external agent

    def constrain_actions(self, goal : int = None) -> List[int]:
        """
        THis will return the set of actions that might be valid for a given goal but for the momet to increase complexoty we are not constraining the possible actions for each goal
        :param goal: The int goal value that is required
        :return: list of int containint possible actions for the goal
        """
        if goal is None:
            pass
        else:
            pass
        return list(range(self.primitive_action_space))

    def step(self, action):
        #print("Env Controller Step")
        # done = False
        goal_reached = False
        new_state = np.copy(self.current_slot_state)  # copy the state of the current slot state
        current_intent = self.goal
        #print("Step : Current Intent : {}".format(current_intent))
        reward = 0
        if action == 19:
            """
            if the terminating action is picked
            if all slots are covered above the threshold then award otherwise penalize
            """

            if self.check_confidence_state(current_intent):
                # give full reward
                reward = self.w2 * self.calculate_external_reward(np.zeros(self.slot_space_size),
                                                                  self.current_slot_state, current_intent)
            else:
                reward = -self.w2 * self.calculate_external_reward(self.current_slot_state,
                                                                   np.ones(self.slot_space_size), goal=current_intent)
            # shifting the below part in the meta step acitons
            # self.current_intent_no += 1  # if all the intents in the current object are over

            goal_reached = True
            return self.current_slot_state, reward, goal_reached,

        # if not terminating action
        relevant_actions: List[int] = impdicts.intent2action[current_intent]
        if action not in relevant_actions:
            reward = -self.w1
        else:
            if action in impdicts.askActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2 * random.random() + 0.55
                # pass # here the action will be same as the slot number
            elif action in impdicts.reaskActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    if new_state[slot] < 0.1:
                        pass
                    else:
                        new_state[slot] = (1 - new_state[slot]) * 0.85 + new_state[slot]
                # pass # Use the index of the list to
            elif action in impdicts.hybridActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2 * random.random() + 0.55
                # pass # The hybrid action
            else:
                print("Wrong action picked up please see the system.\nExiting.....")
                sys.exit()
                # pass # put an error message in the same
            # calculate the reward
            reward = self.w2 * self.calculate_external_reward(self.current_slot_state, new_state,
                                                              current_intent) - self.w1


        self.current_slot_state = np.copy(new_state)
        return self.current_slot_state, reward, False


# This Enviornment is meant to have multiple intents at a given time , hence have 6 options
class MetaEnvMulti:

    def __init__(self, w1 : float=  1, w2 : float = 8, w3 : float = 13,   intent_space_size : int = 5, slot_space_size : int  =  8, options_space : int = 6, primitive_action_space : int = 20 ):
        """
        Args
        :param intent_space_size: The number of intents in the system
        :param slot_space_size: The number of slot that need to be filled with the probability values
        :param options_space: The number of available options in the meta policy
        :param primitive_action_space:  The number of primitve actions available
        :param current_intents: Array containing the active intents for the current obj
        """
        self.threshold = 0.7
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3 # probably will not be using this
        self.intent_space_size = intent_space_size
        self.slot_space_size = slot_space_size
        self.options_space = options_space
        self.primitive_action_space = primitive_action_space
        self.current_obj_intent =[] # This shows all the intents that have to be served in this object
        self.slot_states = []
        self.current_intent_state= []
        self.current_slot_state = np.array([])
        self.current_intent_group_no = 0 # Intent group being served
        self.total_intent_group_nos = 0 # This will contain the total intent groups, or the amount of times the user_agnet action should ideally be invoked
        self.no_intents = 0 # the number of intents to be served i.e. len(self.current_obj_intent)
        self.goal_iter = [] # The number of iterations done for each subgoal completion
        self.current_obj_intent_groups = [] # A 2D list, containting list of the itnetns to be served at a given group time
        self.reset()
        self.latest_start_confidence_start = [] # Store the last confidnce state before the start of the option play for the subpolicy
        # self.slot_states # this is the list of all the slot states encountered in runs
        # self.current_slot_state # this is state of the confidence values in the current context
        # self.current_intent_state # This contains the current intent that the env is focusing on (currently a one hot vector later, maybe changed to a composite of multiple intents
        # self.intent_states # collection of the timeline of the intent states encountered



    def reset(self, prob_random_start : float = 0.5): # just trying this return syntax
        """
        confidence_state, intent_state = env.reset()
        We need to init the object with a set of new intents

        :return: confidence_state : The slot state with confidence  values
        intent_state : The intent state for the current intent
        # Status : Completed, not tested
        """
        self.current_obj_intent = []
        self.no_intents = np.random.randint(1,self.intent_space_size + 1) # Produces a number between 1 and 5 for the number of intents
        # get a starting number of intents from 0 to intent_space
        temp_intents = list(range(0, self.intent_space_size))
        for iter in range(self.no_intents):
            indx = np.random.randint(0, len(temp_intents))
            self.current_obj_intent.append(temp_intents[indx])
            del temp_intents[indx]
        # after these steps teh current_onbj_intent contains the intents and there order to be followred in sccheduling intents for the agent
        if random.random() < prob_random_start:
            # do a random init
            self.random_state_init()
        else:
            self.state_init()
        self.current_obj_intent_groups = self.create_intent_group()
        # now we will set the intital intent state of the system and also the buffer to store the intent states
        self.current_intent_group_no = 0 # Keeps track of the intent number being served
        self.intent_states = np.array([ utils.multi_hot(self.current_obj_intent_groups[self.current_intent_group_no], self.intent_space_size)]) # setting the starting intent

        # self.intent_states = np.array([ util.one_hot( self.current_obj_intent[self.current_intent_no], self.intent_space_size)]) # setting the starting intent
        self.current_intent_state = self.intent_states[-1]
        return [self.current_slot_state, self.current_intent_state]

    def create_intent_group(self, current_obj_intent = None):
        """
        This returns a list of list by grouping the intents in current_obj_intent
        :param current_obj_intent:
        :return: a List(list())
        Testing :  Done
        """
        if current_obj_intent is None:
            current_obj_intent = self.current_obj_intent.copy()
        no_intents_left = len(current_obj_intent)
        # generate a random number between 1 and no_intent_left and group them
        group_intents = []
        while no_intents_left > 0:
            current_group = []
            current_group_intent_no = np.random.randint(1, no_intents_left + 1)
            no_intents_left -= current_group_intent_no
            current_group = current_obj_intent[:current_group_intent_no]
            current_obj_intent = current_obj_intent[current_group_intent_no:]
            group_intents.append(current_group)
        return group_intents


    def random_state_init(self):
        self.slot_states = np.array([[random.random() for i in range(self.slot_space_size)]])
        self.current_slot_state = self.slot_states[-1]

    def state_init(self):
        self.slot_states = np.zeros((1, self.slot_space_size))  # initliase with zero for all values
        self.current_slot_state = self.slot_states[-1]

    def controller_step(self, goal, action):
        """

        :param goal: The goal the controller is fulfilling to appropriately reward it
        :param action: : The action that the agent is taking pertaining to that goal
        :return: next_confidence_state, reward, goal_completed
        """
        # print("Env Controller Step")
        # Explicitytly writing the goal condition
        if goal == 5:
            # we don't need to do any thing for this and probably this will never Occur
            print("IMPLEMENTATION ERROR : This condition should have never been approached")
            sys.exit()
        # done = False
        goal_reached = False
        new_state = np.copy(self.current_slot_state)  # copy the state of the current slot state
        current_intent = goal
        # print("Step : Current Intent : {}".format(current_intent))
        reward = 0
        if action == 19:
            """
            if the terminating action is picked
            if all slots are covered above the threshold then award otherwise penalize
            """

            if self.check_confidence_state(current_intent):
                # give full reward
                reward = self.w2 * self.calculate_external_reward(np.zeros(self.slot_space_size),
                                                                  self.current_slot_state, current_intent)
            else:
                reward = -self.w2 * self.calculate_external_reward(self.current_slot_state,
                                                                   np.ones(self.slot_space_size), goal=current_intent)
            # shifting the below part in the meta step acitons
            # self.current_intent_no += 1  # if all the intents in the current object are over

            goal_reached = True
            return self.current_slot_state, reward, goal_reached,

        # if not terminating action
        relevant_actions: List[int] = impdicts.intent2action[current_intent]
        if action not in relevant_actions:
            reward = -self.w1
        else:
            if action in impdicts.askActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2 * random.random() + 0.55
                # pass # here the action will be same as the slot number
            elif action in impdicts.reaskActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    if new_state[slot] < 0.1:
                        pass
                    else:
                        new_state[slot] = (1 - new_state[slot]) * 0.85 + new_state[slot]
                # pass # Use the index of the list to
            elif action in impdicts.hybridActions:
                slots = impdicts.action2slots[action]
                for slot in slots:
                    new_state[slot] = 0.2 * random.random() + 0.55
                # pass # The hybrid action
            else:
                print("Wrong action picked up please see the system.\nExiting.....")
                sys.exit()
                # pass # put an error message in the same
            # calculate the reward
            reward = self.w2 * self.calculate_external_reward(self.current_slot_state, new_state,
                                                              current_intent) - self.w1  # get the reward for increase in confidecne and the decrrease in iteration
            # Although the calculate external reward is not required as we are already cross checking the things in the first if condition

        self.current_slot_state = np.copy(new_state)
        return self.current_slot_state, reward, False

    def meta_step_start(self,option):
        """
        This will be responsible for storing the current_confidence-state, to make a comparison with the same in the future
        :param option: Take as intput the option chosed by the meta policy
        :return: return the current confidence state of the dialogue
        """
        # store the current_confidence state

        self.latest_start_confidence_start = np.copy(self.current_slot_state)
        return self.latest_start_confidence_start

    def meta_step_end(self, option):
        done = False
        reward = 0
        print("The option picked up is : {}".format(option))
        if option == 5: # we need to switch to the next set of intents
            # We Will check if it has filled all the relevant slots for the intents and reward it appropritaley for each slot filled

            # Implementing the reward for the meta policy

            # And also modify the next intent state
            intent_groups = len(self.current_obj_intent_groups) # this gives the number of intent grousp that we need to cycel thoufh
            reward = self.user_agent_reward() # THis function will return the reward for the user agnet action,
            self.current_intent_group_no +=1
            # check if we are done with the thing or not. I.e. if we are equal with the number of groups
            # print(self.current_obj_intent_groups)
            # print(self.current_intent_group_no)
            # self.intent_state = utils.multi_hot(self.current_obj_intent_groups[self.current_intent_group_no], 5)
            if self.current_intent_group_no >= intent_groups:
                done = True
            else:
                self.current_intent_state = utils.multi_hot(self.current_obj_intent_groups[self.current_intent_group_no], 5)
                # dont change the intente state

            return self.latest_start_confidence_start, self.current_slot_state, self.current_intent_state, reward, done
            # TODO we also need to penalize the agent for extra steps that it takes to acheive a certain task because, it should pick the most effecient process
        else:
            # FIXME  : 19.2.19
            # TODO , currently the wrong option is not being penzliaed , we have to implmemnt to change the intent space whrn the option is picked and that intent has to be remoced gtom the state and for a wrong option we should give
            reward = self.calculate_external_reward(np.copy(self.latest_start_confidence_start), self.current_slot_state, option)
            # the current_intent_state should not change
            return self.latest_start_confidence_start, self.current_slot_state, self.current_intent_state,reward, done


    def meta_step_end2(self, option):
        """
        This works on the following modes
        1. intent-state-mod2
        2. meta-reward-2
        3. meta-state-1
        :param option:
        :return: The updated intent staet, after remobing option from it and the appropriate reward
        """
        done = False
        reward = 0
        print("The option picked up is : {}".format(option))
        if option == 5:
            # TODO Changes from first : ( we check if there are active intents , if yes we penalize the agent with -w2
            if np.sum(self.current_intent_state) > 0.01 :
                reward = -self.w2
            else:
                reward = self.user_agent_reward()
            intent_groups = len(self.current_obj_intent_groups) # this gives the number of intent grousp that we need to cycel thoufh

            self.current_intent_group_no +=1
            if self.current_intent_group_no >= intent_groups:
                done = True
            else:
                self.current_intent_state = utils.multi_hot(self.current_obj_intent_groups[self.current_intent_group_no], 5)
                # TODO change the intenet stat

            return self.latest_start_confidence_start, self.current_slot_state, self.current_intent_state, reward, done
        else:
            # FIXME  : 19.2.19
            # TODO , currently the wrong option is not being penzliaed , we have to implmemnt to change the intent space whrn the option is picked and that intent has to be remoced gtom the state and for a wrong option we should give
            # self.current_intent_state contains all the relevant intents
            reward = self.calculate_external_reward(np.copy(self.latest_start_confidence_start), self.current_slot_state, option)
            # the current_intent_state should not change
            return self.latest_start_confidence_start, self.current_slot_state, self.current_intent_state,reward, done

    def user_agent_reward(self):

        """
        We will requrie teh use of
        self.current_slot_state
        self.current_intent_state
        We will return the matching slots for the intetnts specifficed bt current intent state and award the policy for filling all those slots
        :return: A scalar reward value
        """
        relevant_slots = []
        for each_goal in self.current_obj_intent_groups[self.current_intent_group_no]:
            relevant_slots = np.concatenate([relevant_slots, impdicts.intent2slots[each_goal]])
        relevant_slots = list(set(np.array(relevant_slots, dtype=np.int32)))
        # these are the relevant slots checl the value for these
        correct =  all(self.current_slot_state[relevant_slots] > self.threshold)
        if correct:
            # give reward for all the slots
            # return self.w1*
            return self.w2*np.sum(self.current_slot_state[relevant_slots])
        else:
            return -self.w2*(float(len(self.current_slot_state[relevant_slots])) - np.sum(self.current_slot_state[relevant_slots])) # Subtract the remaining confidence values of the slots that is requried to fill the same.
    #
    def user_agent_reward2(self):
        '''
        Implementation of the user rewward function which panelizes the fiollling the filling of wrong slots.
        Variables Introduced : self.starting_slot_state_intent_group : This will contain the confidence values of the starting slot state when beginning the given intent group
        :return:
        '''

    def meta_step_end_legacy1(self, option) -> Tuple[int, float, bool]  :
        """

        :param option: The option chosen
        :return: return the  next intent, reward, done (still skeptical about returning the confidence state)
        Points :
        This function will funciton normally for all options except user_agent (5) option, for that we need to check if the all the group intents are served or not
        """
        done = False
        reward = 0
        # return the next sets of intents
        current_intent = self.current_obj_intent[self.current_intent_no]
        reward = self.w2*self.calculate_external_reward(np.copy(self.latest_start_confidence_start), self.current_slot_state, current_intent)
        self.current_intent_no += 1  # if all the intents in the current object are over
        new_intent = current_intent
        if self.current_intent_no >= self.no_intents:
            done = True
        else:
            self.current_intent_state = utils.one_hot(
                self.current_obj_intent[self.current_intent_no], self.intent_space_size)
            new_intent = self.current_obj_intent[self.current_intent_no]
        return self.current_intent_state, reward, done

    def calculate_external_reward(self, start_state : np.ndarray, goal_state : np.ndarray, goal : int) -> float :
        """
        This function is supposed to check the goal and then check the progress
        in the confidence values of the slots positions with respect to that goal
        :param start_state: The starting state of the confidence values of options
        :param goal_state: The ending state of the confidence values of the options
        :param goal: The goal currently working for
        :return: return the reward
        """
        relevants_slots = impdicts.intent2slots[goal] # this will return the slots to be measured
        # now we will calculate the differen from both
        diff_confidence: float = np.sum(goal_state[relevants_slots] - start_state[relevants_slots])
        # now we can multiply by weights
        return diff_confidence # or we can keep and differnt weight factor for the external agent


    def check_confidence_state(self, goal):
        """
        THis function will check all the slots pertaining to the goal and see if its satisfies the threshold, if it does so , it returns a True value, otherwise it returns a false value.
        :param goal: The goal that its currently checking for
        :return: True/False
        Rules & Points :
        1. I am using the threshold currently to analyze
        """
        slots_to_be_checked = impdicts.intent2slots[goal] # these are the slots
        return all(self.current_slot_state[slots_to_be_checked] > self.threshold)

    def constrain_options(self):
        '''
        THis is only a dummy function and just returns the list of options
        :return:
        '''
        return list(range(self.options_space))

