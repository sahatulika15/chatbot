import argparse




def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cw','--controller-weights',required = False, help = "Weight for the controller policy model (i..e Mention  the file from whcih we need to load the qweights)")
    # add the training parameters
    ap.add_argument('-e','--episodes',required = False ,help='The number of episodes that the Meta Policy should train for.) If not specified then a default method will be used to save the file', type = int,default = 1_00_000) # default is 1 lakh
    ap.add_argument('-mw','--meta-weights', required=False, help = "The Weights of the meta policy to be used for training (i.e. the file tobe saved for the same")

    ap.add_argument('-gpu','--set-gpu', required=False,help ='Give the GPU number, if GPU is to be used, otherwise dont mention for run on CPU', type = int)
    ap.add_argument('-bs','--batch-size',required = False, help = "THe Batch size of the required trianing of the meta policy", type= int, default = 64)
    ap.add_argument('-lr','--learning-rate',required=False, default=0.05,type = float, help= 'The learning rate for meta policy training')
    ap.add_argument('-df','--discount-factor',required = False, default = 0.7, type = float , help = 'THe Discount factor for the learning of meta polciy')
    ap.add_argument('-do','--dropout',required = False,default = 0.00, type = float, help = 'The dropout probabliltiy for the meta policy training')

    ap.add_argument('-ch', '--controller-hidden', required = False, help = "The Hidden Layers  configuration of the Controller Policym This portion is not yet implemeneted in the Code \nFormat : n1_n2_n3 ..  , tells n1 nodes in layer 1 , n2 nodes in layer 2 , n3 nodes in layer 3 and so on." , default = '75')
    ap.add_argument('-ns','--number-slots',required=False , help = 'The number of confidence slots in the domain', type = int, default = 8)
    ap.add_argument('-ni','--number-intents',required = False, help = 'The Number of intents that the domain has', type = int, default = 5)
    ap.add_argument('-no','--number-options',required=False, help='The number of options for the meta policy', type = int, default = 6)
    ap.add_argument('-ca','--controller-action', required = False, help = 'The Number of Controller Actions, although thte code is designed to work for only 20 action as of now', type = int, default = 20 )
    ap.add_argument('-p','--save-folder',required = False, help = 'Provide the path to the fodler where we need to store the meta policy',type = str, default = './save/')
    ap.add_argument('-bcl','--break-controller-loop',help = 'Sometimes the Controller POlicy tends to be stuck in infiint loop, to remedy this we will restrict its runs to some value', type = int, default = 100)
    # ap.add_argument('-ln','--loadname',required=False, help = "Give the file to be loaded in the meta policy if yoiu wish to continue training of teh model from a certain point", type = bool , default = )
    ap.add_argument('-nf','--note-file', required = False, help = 'Give a special note that might be needed to add at the end of the file name id the user is not providing the dfile name', default= '')
    ap.add_argument('-cf','--config-folder',required= False, help = 'Give the folder were we should save the config file ', default = './config/')
    # TODO need to enable the facility of the loadname and SaveIn fcaility in the DQN1 program
    ap.add_argument('-lf','--create-log',required = False, help = "Specify if requried to create a log file of the running experiments[yet to be implemented]", default = False)
    args  = vars(ap.parse_args())
    return args
