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
from envs.metaenvmulti import MetaEnvMulti
from NLU_model import NLU



class NLU_Simulator(MetaEnvMulti):
    
    def __init__(self, nlu_name  = 'nlu.h5'):
        '''
        nlu_name : Provide the name of the NLU model to be loadade
        '''
        super().__init__()
        self.nlu_name = nlu_name
        self.nlu_name = NLU(filename = self.nlu_name, load= True)
        self.guessed_slot_values = {
                                    "$ACITY$"   : "Unknown",
                                    "$DCITY$"   : "Unknown",
                                    "$DATE$"    : "Unknown",
                                    "$RTRIP$"   : "Unknown",
                                    "$CLASS$"   : "Unknown",
                                    "$TIME$"    : "Unknown",
                                    "$CITY$"    : "Unknown",
                                    "$TTYPE$"   : "Unknown"
                                    }

        self.tags2position = {     
                                    "$ACITY$"   : 1,
                                    "$DCITY$"   : 0,
                                    "$DATE$"    : 3,
                                    "$RTRIP$"   : 5,
                                    "$CLASS$"   : 4,
                                    "$TIME$"    : 2,
                                    "$CITY$"    : 6,
                                    "$TTYPE$"   : 7
                            }
        self.position2tags = {
                                    1:"$ACITY$",
                                    0:"$DCITY$",
                                    3:"$DATE$" ,
                                    5:"$RTRIP$",
                                    4:"$CLASS$",
                                    2:"$TIME$" ,
                                    6:"$CITY$" ,
                                    7:"$TTYPE$"
                             }

    def step(self, action):
        '''
        Redefine the action to be performed
        '''
    

    def step(self,action):
        done = False

        reward = 0
        # action is a numeric value
        # TODO replace tags in reply with the real values

        # fixmed this part will come at the end because reply can change


        # print the user reply
        new_state = self.current_state.copy()
        if action == 13: # terminating action # todo dont update the state and done = true and give the reward comapre the guessed and real slots and decide a reward function
            done = True

            # fixme decide the reward over here
            # fixme still the reward is left to decide
            # dont change the new_state
            # reward = ???
            reply = random.choice(airfare.indx2user_replies[action])
            # check if the implementation is correct or not
            all_correct = 1
            for i in range(5):
                if self.guessed_slot_values[self.position2tags[i]] != self.real_slots_values[self.position2tags[i]]:
                    all_correct = 0
            if all_correct == 1:
                reward = 1
            else:
                reward = -1


        elif action > 7 and action <= 12: # reask actions
            slot = action - 8 # give the slot number
            if self.guessed_slot_values[self.position2tags[slot]] == self.real_slots_values[self.position2tags[slot]]:
                reply = "yes"
                new_state[slot] = 1
            else:
                reply = "no"
                new_state[slot] = 0

        else: # action 0 to 7
            # [pred,prob] = self.nlu_model.parse_sentence(reply)
            reply = random.choice(airfare.indx2user_replies[action])
            for k in self.real_slots_values:
                if k in reply:
                    reply = reply.replace(k, self.real_slots_values[k])

            tags = defaultdict(list)
            prob_values = defaultdict(list)
            ourtags = []
            [pred, prob] = self.nlu_model.parse_sentence(reply)
            for t in pred:
                ourtags.append(airfare.labels2labels[t])

            print (ourtags)
            for i, val in enumerate(ourtags):
                if val in self.guessed_slot_values:
                    # guessed_slots_values[val] = reply.split(" ")[i]
                    # new_state[positions[val]] = prob[i]
                    tags[val].append(reply.split(" ")[i])
                    prob_values[val].append(prob[i])
            for k in prob_values:
                prob_values[k] = [sum(prob_values[k]) / len(prob_values[k])]

            # print tags
            # print prob_values
            for k in tags:
                self.guessed_slot_values[k] = ' '.join(tags[k])
                new_state[self.positions[k]] = prob_values[k][0]


            # finished else
        agent_sentence = airfare.actions_sentences[action]
        # reply = random.choice(airfare.indx2user_replies[action])
        for k in self.guessed_slot_values:
            if k in agent_sentence:
                agent_sentence = agent_sentence.replace(k, self.guessed_slot_values[k])

        print("\n\nAgent >{}".format(agent_sentence))
        print("User  >{}\n\n".format(reply))

        # calculate the apropiate reward
        # reward = 0
        self.current_state = np.array([float('%.3f' % elem) for elem in new_state])
        self.states = np.append(self.states, [self.current_state], axis=0)
        return [self.current_state,reward,done, reply]
