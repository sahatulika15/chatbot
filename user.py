import random
import numpy as np
from util import impdicts


class User:
    '''
    THis class will be used to simualte a user and his/her behvaiour
    '''
    def __init__(self):
        # init with a set of slots for each one
        self.slot_values = {}
        self.intent_groups = [] # this will contain the list of group of intent that the user would want to ask
        self.reset()



    def reset(self):
        '''
        Reset the diffrent values
        :return:
        '''
        for t in impdicts.all_tags:
            self.slot_values[t] = random.choice(impdicts.tags2values[t])
        # this will randomly assign the tags to the values

    def listen_reply(self, sentence, action, guessed_slots):
        '''
        This will take input a sentence and give the next output
        :param sentence:
        :param action:
        :param guessed_slots:
        :return:
        '''
        reply = ""
        if action in range(0,11):
            # answer the question
            pass
        elif action in range(11, 19):
            # give the feedback for reask by comparing the gueesed vs real
            tag = impdicts.position2tags[action - 11]

            if guessed_slots[tag] == self.slot_values[tag]:
                reply = impdicts.actions2replies[1]


        elif action == -1 :
            # This is the -1 action where teh user will shift  to a different intent or end the converstion
            pass

        return reply






