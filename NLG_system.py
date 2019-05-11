tag2values = {
"$CITY$":["boston","pittsburgh","washington","tacoma"],
"$TTYPE$":["car","limousine"],
"$DCITY$":["boston","pittsburgh","washington","tacoma"],
"$ACITY$":["denver","baltimore","orlando","philadelphia"],
"$DATE$":["twenty eight december","fourth january"],
"$RTRIP$":["round","one way"],
"$CLASS$":["business","economy","first"], 
"$TIME$" :["evening","morning","afternoon"],
}

agent_action2sentences = {
    -1  :["Hello How can I help you", "Welcome to flight enquiry system"],
    0   :["Please tell the city of Departure?"],
    1   :["Please tell the city of Arrival?"],
    2   :["Please tell the time of departure"],
    3   :["Please tell the date of departure"],
    4   :["Please specify the class of flight"],
    5   :["Round trip or one way fare?"],
    6   :[ "Please tell the city of ground transportation?"],
    7   :["Please tell the transport type of service?"],
    8   :[ "From where to where"],
    9   :[ "Specify the time and date"],
    10  :["Please specify the place and transport type of the ground service"],
    11  :["Are travelling from $DCITY$?"],
    12  :["Are you travelling to $ACITY$?"],
    13  :["Are you travelling at $TIME$"],
    14  :["Are you travelling on $DATE$"],
    15  :["You would like to travel via $CLASS$"],
    16  :["Do you want $RTRIP$ fares?"],
    17  :[ "You need ground service in $CITY$?"],
    18  :["You need $TTYPE$ service?"],
    19  :["Okay move onto the next intent", "Done with this"]

}

user_action2sentences = {
    -1  :["I want to travel from $DCITY$ to $ACITY$ on $DATE$ at $TIME$","I want to travel from $DCITY$ to $ACITY$ on $DATE$","I want to travel from $DCITY$ to $ACITY$","I want to travel to $ACITY$","flights from $DCITY$ to $ACITY$ on $DATE$ at $TIME$","flights from $DCITY$ to $ACITY$ on $DATE$","flights from $DCITY$ to $ACITY$"], # need to change this to have senteces, with multiple intents
    0   :["I want to travel from $DCITY$","from $DCITY$","$DCITY$","flights from $DCITY$"],
    1   :["I want to travel to $ACITY$","to $ACITY$","to $ACITY$","flights to $ACITY$"],
    2  :["at $TIME$","around $TIME$","flights at about $TIME$"],
    3  :["I want to travel on $DATE$","on $DATE$","need flights for $DATE$"],
    4  :["I want to travel in $CLASS$ class","$CLASS$ class","$CLASS$"],
    5  :["I want a $RTRIP$ trip","$RTRIP$ trip","$RTRIP$ trip fare"],
    6  :["within $CITY$","in $CITY$","from the $CITY$"],
    7  :["$TTYPE$ service","$TTYPE$"],
    8  :["I want to travel from $DCITY$ to $ACITY$","flights from $DCITY$ to $ACITY$"],
    9  :["Need to travel on $DATE$ at $TIME$","On $DATE$ at $TIME$","flights on $DATE$ at $TIME$"],
    10  :["$TTYPE$ service in $CITY$","within $CITY$ using $TTYPE$ service"],
    11  :["Yes","Yes please"],
    12  :["Yes","Yes please"],
    13  :["Yes","Yes please"],
    14  :["Yes","Yes please"],
    15  :["Yes","Yes please"],
    16  :["Yes","Yes please"],
    17  :["Yes","Yes please"],
    18  :["Yes","Yes please"],
    19  :[]

}