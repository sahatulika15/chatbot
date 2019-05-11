import random
from chatbot import Chatbots, UserSimulator
from user import User
import argparse
from util import impdicts

ap = argparse.ArgumentParser()

ap.add_argument('-m', '--mode', default = "simulation", help = "Enter the mode of opearation for the chatbot")


args  = vars(ap.parse_args())

#print(args)



def interact_human():
    '''
    This will be used to interact with a human
    :return:
    '''
    cbot = Chatbots()
    user = UserSimulator()
    action = 0
    response = random.choice(impdicts.start_user_replies)
    prev_action = 0
    count = 0
    print(response)
    while action != -2:
        #reply = input(">")
        [reply, action, slots] = cbot.step(response)
        if(prev_action==action):
            count+=1
        else:
            prev_action = action
        print("Action {} : {} ".format(action, reply))
        if(action==-2 or count > 5):
            break
        elif(action != -1):
            response = user.response(action,slots)
        else:
            response = user.new_intent()
        print(response)

    # done conversation
'''
def interact_simulation():
    
    This will playout a simulationf for the uset
    :return:
    
    cbot = Chatbots()
    action = -3
    while action != -2:
        if action == -3 : # this means the greeing action

'''

def main():
    print("in main")
    mode = args['mode']
    if mode == "human":
        print("in main")
        interact_human()
    else:
        pass



if __name__ == "__main__":

    main()
