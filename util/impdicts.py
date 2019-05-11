from typing import Tuple, List, Dict


slot2tag = {'to_loc' : '$ACITY$','froms_loc' : '$DCITY$',
            'date' : '$DATE$','time' : '$TIME$',
            'class' : '$CLASS$', 'round_trip': '$RTRIP$' ,
            'city_name' : '$CITY$', 'transport_type' :'$TTYPE$'}

tag2slot = {'$ACITY$' : 'to_loc','$DCITY$': 'from_loc',
            '$DATE$' : 'date','$TIME$': 'time',
           '$CLASS$':  'class', '$RTRIP$': 'round_trip' ,
           '$CITY$' : 'city_name', '$TTYPE$' : 'transport_type' }




intent_tags = {
      'flight' : ['$ACITY$','$DATE$','$CLASS$','$DCITY$','$TIME$' ],
      'airfare' : ['$ACITY$','$DATE$','$CLASS$','$RTRIP$',
            '$DCITY$'],
      'airline' : ['$ACITY$','$CLASS$','$DCITY$'],
      'ground_service' : ['$CITY$', '$TTYPE$'],
      'ground_fare' : ['$CITY$', '$TTYPE$']
}

intent2indx =  {'flight': 0,'airfare' : 1,'airline' :2, 'ground_service' : 3, 'ground_fare' : 4}


indx2intent = {0:'flight',1:'airfare',2:'airline',
               3: 'ground_service', 4:'ground_fare'}
               

# Dictionaries
## DIctionary which maps intent to action space
intent2action ={ 0:[0, 1, 2, 3, 4, 8, 9, 11, 12, 13, 14, 15, 19], # Intent Flight
                    1:[ 0, 1, 3, 4, 5, 8, 11, 12, 14, 15, 16, 19], # Intent Airfare
                    2:[ 0, 1, 4, 8, 11, 12, 15, 19], # Intent airline
                    3:[ 6, 7, 10, 17, 18, 19], # Intent Ground Service
                    4:[ 6, 7, 10, 17, 18, 19] # Intent Ground Fare
}

# mapping of the intent to there relevant slots values
intent2slots = {0 : [0,1,2,3,4],
                1 : [0, 1, 3, 4, 5],
                2 : [0, 1, 4,],
                3 : [ 6, 7,],
                4 : [ 6, 7,],
                }

# This is kinda an overkill
action2slots = { 0:[0],
                 1: [1],
                 2: [2],
                 3: [3],
                 4: [4],
                 5: [5],
                 6: [6],
                 7: [7],
                 8: [0,1],
                 9: [2,3],
                 10: [6,7],
                 11: [0],
                 12: [1],
                 13: [2],
                 14: [3],
                 15: [4],
                 16: [5],
                 17: [6],
                 18: [7] }


# we can use dictionarites also
askActions = [0,1,2,3,4,5,6,7]
hybridActions = [8,9,10]
reaskActions = [11,12,13,14,15,16,17,18]

tags2values = {"$DCITY$":["boston","pittsburgh","washington","tacoma"],
"$ACITY$":["denver","baltimore","orlando","philadelphia"],
"$TIME$":["evening","morning","afternoon"],
"$RTRIP$":["round","one way"],
"$DATE$":["twenty eight december","fourth january"], # removed ,"fifteenth june"
"$CLASS$":["business","economy","first"],
"$CITY$":["boston","pittsburgh","washington","tacoma"],
"$TTYPE$":["car","limousine"]
}

start_sentences_agent = ["Hello How may I help you"]
# these sentences will act as teh conversatoin starts
intent_sentences = ["Show me the cheapest fares","show me the flights",
                    "I would like information about ground transportation",
                    "Show me flights to $ACITY$", "Flights from $DCITY$","I would like to travel from $DCITY$ to $ACITY$"
                    ]


start_actions_sentences = ["Hello How can I help you", "Welcome to flight enquiry system"]

user_replies = ["I would like information about ground transportation", "Show me the list of airline as well"
                    "Show me the ground transportation fare", "No nothing"]
                    
start_user_replies = ["give me the flights and fares from $DCITY$","show me all flights and fares from $DCITY$", "what ground transportation is available from the $CITY$ and how much does it cost"]


intent_replies = {
      'ground_fare' : ["No Thanks","show me the flights","Show me the list of airline","Show me the cheapest fares"], 
      'ground_service' : ["No Thanks","Show me the ground transportation fare","show me the flights","Show me the list of airline","Show me the cheapest fares"], 
      'airfare' : ["No Thanks","what about the ground transportation", "Show me the list of airline as well"],
      'flight' : ["No Thanks", "Show me the cheapest fares","what about the ground transportation" ],
      'airline': ["No Thanks","Show me the cheapest fares","show me the flights","What about ground service"]
}


# have to see the usability of this
intents = ['flight','airfare','airline','ground_service','ground_fare']

intents2indx =  {'flight': 0,'airfare' : 1,'airline' :2, 'ground_service' : 3,
            'ground_fare' : 4}

all_tags = ['$DCITY$', '$ACITY$','$TIME$','$DATE$','$CLASS$','$RTRIP$', '$CITY$','$TTYPE$','$NULL$']

true_tags = ['$DCITY$', '$ACITY$','$TIME$','$DATE$','$CLASS$','$RTRIP$', '$CITY$','$TTYPE$']


tags2position = {     
                  "$ACITY$"   : 1,
                  "$DCITY$"   : 0,
                  "$DATE$"    : 3,
                  "$RTRIP$"   : 5,
                  "$CLASS$"   : 4,
                  "$TIME$"    : 2,
                  "$CITY$"    : 6,
                  "$TTYPE$"   : 7
                            }
position2tags = {
                  1:"$ACITY$",
                  0:"$DCITY$",
                  3:"$DATE$" ,
                  5:"$RTRIP$",
                  4:"$CLASS$",
                  2:"$TIME$" ,
                  6:"$CITY$" ,
                  7:"$TTYPE$"
                             }

# I am yet to anme teh realtive things properly like 
# 'B-depart_date.today_relative'
# is currently null whereas it tells us about the relative time for today etc"tomorrow"
labels2labels={'B-aircraft_code':all_tags[8],
 'B-airline_code':all_tags[8],
 'B-airline_name':all_tags[8],
 'B-airport_code':all_tags[8],
 'B-airport_name':all_tags[8],
 'B-arrive_date.date_relative':all_tags[8],
 'B-arrive_date.day_name':all_tags[8],
 'B-arrive_date.day_number':all_tags[8],
 'B-arrive_date.month_name':all_tags[8],
 'B-arrive_date.today_relative':all_tags[8],
 'B-arrive_time.end_time':all_tags[8],
 'B-arrive_time.period_mod':all_tags[8],
 'B-arrive_time.period_of_day':all_tags[8],
 'B-arrive_time.start_time':all_tags[8],
 'B-arrive_time.time':all_tags[8],
 'B-arrive_time.time_relative':all_tags[8],
 'B-booking_class':all_tags[4],
 'B-city_name':all_tags[6],
 'B-class_type':all_tags[4],
 'B-compartment':all_tags[8],
 'B-connect':all_tags[8],
 'B-cost_relative':all_tags[8],
 'B-day_name':all_tags[8],
 'B-day_number':all_tags[8],
 'B-days_code':all_tags[8],
 'B-depart_date.date_relative':all_tags[8], # converted from 2 -> 8
 'B-depart_date.day_name':all_tags[3],
 'B-depart_date.day_number':all_tags[3],
 'B-depart_date.month_name':all_tags[3],
 'B-depart_date.today_relative':all_tags[8], # converted from 2 -> 8
 'B-depart_date.year':all_tags[3],
 'B-depart_time.end_time':all_tags[2],
 'B-depart_time.period_mod':all_tags[2],
 'B-depart_time.period_of_day':all_tags[2],
 'B-depart_time.start_time':all_tags[2],
 'B-depart_time.time':all_tags[2],
 'B-depart_time.time_relative':all_tags[2], # converted from 3 -> 8
 'B-economy':all_tags[4],
 'B-fare_amount':all_tags[8],
 'B-fare_basis_code':all_tags[8],
 'B-flight':all_tags[8],
 'B-flight_days':all_tags[8],
 'B-flight_mod':all_tags[8],
 'B-flight_number':all_tags[8],
 'B-flight_stop':all_tags[8],
 'B-flight_time':all_tags[8],
 'B-fromloc.airport_code':all_tags[8],
 'B-fromloc.airport_name':all_tags[8],
 'B-fromloc.city_name':all_tags[0],
 'B-fromloc.state_code':all_tags[0],
 'B-fromloc.state_name':all_tags[0],
 'B-meal':all_tags[8],
 'B-meal_code':all_tags[8],
 'B-meal_description':all_tags[8],
 'B-mod':all_tags[8],
 'B-month_name':all_tags[8],
 'B-or':all_tags[8],
 'B-period_of_day':all_tags[8],
 'B-restriction_code':all_tags[8],
 'B-return_date.date_relative':all_tags[8],
 'B-return_date.day_name':all_tags[8],
 'B-return_date.day_number':all_tags[8],
 'B-return_date.month_name':all_tags[8],
 'B-return_date.today_relative':all_tags[8],
 'B-return_time.period_mod':all_tags[8],
 'B-return_time.period_of_day':all_tags[8],
 'B-round_trip':all_tags[5],
 'B-state_code':all_tags[8],
 'B-state_name':all_tags[8],
 'B-stoploc.airport_code':all_tags[8],
 'B-stoploc.airport_name':all_tags[8],
 'B-stoploc.city_name':all_tags[8],
 'B-stoploc.state_code':all_tags[8],
 'B-time':all_tags[8],
 'B-time_relative':all_tags[8],
 'B-today_relative':all_tags[8],
 'B-toloc.airport_code':all_tags[8],
 'B-toloc.airport_name':all_tags[8],
 'B-toloc.city_name':all_tags[1],
 'B-toloc.country_name':all_tags[1],
 'B-toloc.state_code':all_tags[1],
 'B-toloc.state_name':all_tags[1],
 'B-transport_type':all_tags[7],
 'I-airline_name':all_tags[8],
 'I-airport_name':all_tags[8],
 'I-arrive_date.day_number':all_tags[8],
 'I-arrive_time.end_time':all_tags[8],
 'I-arrive_time.period_of_day':all_tags[8],
 'I-arrive_time.start_time':all_tags[8],
 'I-arrive_time.time':all_tags[8],
 'I-arrive_time.time_relative':all_tags[8],
 'I-city_name':all_tags[6],
 'I-class_type':all_tags[4],
 'I-cost_relative':all_tags[8],
 'I-depart_date.day_number':all_tags[3],
 'I-depart_date.today_relative':all_tags[3],
 'I-depart_time.end_time':all_tags[2],
 'I-depart_time.period_of_day':all_tags[2],
 'I-depart_time.start_time':all_tags[2],
 'I-depart_time.time':all_tags[2],
 'I-depart_time.time_relative':all_tags[2],
 'I-economy':all_tags[4],
 'I-fare_amount':all_tags[8],
 'I-fare_basis_code':all_tags[8],
 'I-flight_mod':all_tags[8],
 'I-flight_number':all_tags[8],
 'I-flight_stop':all_tags[8],
 'I-flight_time':all_tags[8],
 'I-fromloc.airport_name':all_tags[8],
 'I-fromloc.city_name':all_tags[0],
 'I-fromloc.state_name':all_tags[0],
 'I-meal_code':all_tags[8],
 'I-meal_description':all_tags[8],
 'I-restriction_code':all_tags[8],
 'I-return_date.date_relative':all_tags[8],
 'I-return_date.day_number':all_tags[8],
 'I-return_date.today_relative':all_tags[8],
 'I-round_trip':all_tags[5],
 'I-state_name':all_tags[8],
 'I-stoploc.city_name':all_tags[8],
 'I-time':all_tags[8],
 'I-today_relative':all_tags[8],
 'I-toloc.airport_name':all_tags[8],
 'I-toloc.city_name':all_tags[1],
 'I-toloc.state_name':all_tags[1],
 'I-transport_type':all_tags[7],
'O':all_tags[8]
}


# this will be the replies that the user will give
actions2replies = {
    0 : ["I want to travel from $DCITY$","from $DCITY$","$DCITY$","flights from $DCITY$"],
    1 : ["I want to travel to $ACITY$","to $ACITY$","to $ACITY$","flights to $ACITY$"],
    2 : ["at $TIME$","around $TIME$","flights at about $TIME$"],
    3 : ["I want to travel on $DATE$","on $DATE$","need flights for $DATE$"],
    4 : ["I want to travel in $CLASS$ class","$CLASS$ class","$CLASS$"],
    5 : ["I want a $RTRIP$ trip","$RTRIP$ trip","$RTRIP$ trip fare"],
    6 : ["within $CITY$","in $CITY$","from the $CITY$"],
    7 : ["$TTYPE$ service","$TTYPE$"],
    8 : ["I want to travel from $DCITY$ to $ACITY$","flights from $DCITY$ to $ACITY$"],
    9 : ["Need to travel on $DATE$ at $TIME$","On $DATE$ at $TIME$","flights on $DATE$ at $TIME$"],
    10 : ["$TTYPE$ service in $CITY$","within $CITY$ using $TTYPE$ service"],
    11 : ["No","Yes"],
    12 : ["No","Yes"],
    13 : ["No","Yes"],
    14 : ["No","Yes"],
    15 : ["No","Yes"],
    16 : ["No","Yes"],
    17 : ["No","Yes"],
    18 : ["No","Yes"],
    19 : ["okay"],
    -1 : ["NOTHING"]
}

# this will be the questions that the agent will ask
# todo
action2questions = {
    0:["Please tell the city of Departure" ],
    1:["Please tell the city of Arrival"],
    2:["Please tell the time of departure"],
    3:["Please tell the date of departure"],
    4:["Please specify the class of flight"],
    5:["Round trip or one way fare"],
    6:["Please tell the city of ground transportation"],
    7:["Please tell the transport type of service"],
    8:["From where to where"],
    9:["Specify the time and date"],
    10:["Please tell the place and transport type of the ground service"],
    11:["Are you travelling to $ACITY$"],
    12:["Are travelling from $DCITY$"],
    13:["Are you travelling at $TIME$"],
    14:["Are you travelling on $DATE$"],
    15:["You would like to travel via $CLASS$"],
    16:["Do you want $RTRIP$ fares"],
    17:["You need ground service in $CITY$"],
    18:["You need $TTYPE$ service"],
    19:["Provide Information related to intent from database"]
}

# todo This dictionary will conatins all the intents indexeds correspoding to the group of intents indentified
intents2index = {}


actions_sentences = ["Please tell the city of Arrival?",  # action : 0 ask arrival city
                     "Please tell the city of Departure?",  # action : 1 ask dept city
                     "Please tell the time of departure",  # action : 2 ask dep time
                     "Please tell the date of departure?",  # action : 3 ask departure day
                     "Please specify the class of flight?",  # action : 4 ask class of travel
                     "Round trip or one way fare?",  # action : 5 ask round trip
                     "Please tell the city of ground transportation?" # action : 6 ask city
                     "Please tell the transport type of service?",  # action : 7 ask transport type
                     "From where to where?",  # action : 8 ask Departure and Arrival City
                     "Specify the time and date",  # action : 9 ask date and time both
                     "Please specify the place and transport type of the ground service",  # action : 10 ask city and transport type
                     "Are you travelling to $ACITY$?",  # action : 11 reask arrival city
                     "Are travelling from $DCITY$?",  # action : 12 reask dept city
                     "Are you travelling at $TIME$",  # action : 13 reask dept time
                     "Are you travelling on $DATE$?",  # action : 14 reask dept day
                     "You would like to travel via $CLASS$?",  # action : 15 reask class
                     "Do you want $RTRIP$ fares?",  # action : 16 reask round trip
                     "You need ground service in $CITY$?",  # action : 17 reask city
                     "You need $TTYPE$ service?",  # action : 18 reask transport type
                     "Here is you itenary\nThe fares of the flights from $DCITY$ to $ACITY$ on $DATE$ and $RTRIP$ trip via $CLASS$ is.\nDo you need anything else?",  # action : 19 close the conversation
]


