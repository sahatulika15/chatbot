"""
This will contain helper functions
Description of the problem
Intents : 
0 : Flight 
1 : Airfare
2 : Airline
3 : Ground Service
4 : Ground Fare
Slots : 
0 : Departure City
1 : Arrival City
2 : Time
3 : Date
4 : Class
5 : Round Trip
6 : Ground City (We can replace this with the arrival city as well)
7 : Transport Type 
Actione : 
0 - 7 : Ask Each Slot (int the above order)
8 - 10 : Hybrid actions
    8 : Ask Dept City and Arr City
    9 : Time and Date
    10 : Ground City and Transport Type
11 - 18 : Reask Each slot value (to confirm if deemed requried)
19 : Terminate the conversation
"""

import numpy as np
import random
from .impdicts import *
import sys

NO_SLOTS = 8
META_STATE_SIZE = 5
META_OPTION_SIZE = 5
CONTROLLER_STATE_SIZE = NO_SLOTS + META_STATE_SIZE
CONTROLLER_ACTION_SIZE = 20




def get_random_action_goal(goal):
    """
    This code for a given goal will return a action (this action will not belong to another goal)
    Args :
    goal : Index specifying the intent
    """
    # check if goal is int , if not get the intent out
    if type(goal) == str: 
        goal = intent2indx[goal]
    return random.choice(intent2action[goal])
    


def one_hot(point, size): 
    '''
    Args: 
    point : The place where we want the one to be
    size : The size of the total vector
    returns : A one dimensional list (i.e. array) of the given number and ots vector
    '''
    vector = np.zeros(size)
    vector[point] = 1.0
    # return np.expand_dims(vector, axis=0) # returns a 2D array
    return vector


def multi_hot(points, size):
    """
    Create a multi hot vector of given size with points values filled. Just a cosmetic name difference for the sake of mkaing
    :param points:
    :param size:
    :return:
    """
    vector = np.zeros(size)
    vector[points]  =1.0
    return vector

def combine_multi_hot(vec1, vec2):
    """
    This will combine 2 multi hot vectors and handle clashes as well
    :param vec1:
    :param vec2:
    :return:
    """
    if len(vec1) != len(vec2):
        print("Mismatch in array size")
        sys.exit()
    return list(np.array((np.array(vec1) + np.array(vec2) ) > 0, dtype = np.int32))


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def printblue(  string):
        print("{}{}{}".format(bcolors.OKBLUE, string, bcolors.ENDC))
    def printgreen(  string):
        print("{}{}{}".format(bcolors.OKGREEN, string, bcolors.ENDC))
    def printwarning(  string):
        print("{}{}{}".format(bcolors.WARNING, string, bcolors.ENDC))
    def printfail(  string):
        print("{}{}{}".format(bcolors.FAIL, string, bcolors.ENDC))
    def printbold(  string):
        print("{}{}{}".format(bcolors.BOLD, string, bcolors.ENDC))
    def printunderline(  string):
        print("{}{}{}".format(bcolors.UNDERLINE, string, bcolors.ENDC))
    def printbold(  string):
        print("{}{}{}".format(bcolors.HEADER, string, bcolors.ENDC))

# print bcolors.WARNING + "Warning: No active frommets remain. Continue?"
#       + bcolors.ENDC
