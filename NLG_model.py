import numpy as np
from util import impdicts
import random
# todo To test this module
class NLG:
    def __init__(self):
        pass

    def generate_sentence(self, controller_action, guessed_slot_values):
        '''
        This will take in controller action and thes guessed slots values to produce a serntence in thr form of reply

        :param controller_action:
        :param slot_valiues:
        :return:
        '''
        sentence = random.choice(impdicts.action2questions[controller_action])
        # fill it with the guessed slots values
        for t in impdicts.true_tags:
            # realpce all the tags with the proper thing
            sentence = sentence.replace(t, guessed_slot_values[t])

        return sentence
