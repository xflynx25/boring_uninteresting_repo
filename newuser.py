# The person running this file needs access to the following
# x this file
# r,w ClientPersonalities
# x ClientOverseer (imports overseer and runs with its personality)

"""
Default chips and deltas files 
(take from past years or this years, and select only up to previous gw)


Check if Folder Setup works in different folder. Make seemless
Interactive Team Input --> human_inputs files
Few things that the user has to put into config (filesystem), actually just put that into private versions, malleable constants
allow to put in their previous point scores, but to allow skip and we use default
"""

### HELLO AND WELCOME TO THE SCRIPT THAT SETS UP A NEW USER'S INFORMATION
### ### HERE, A USER WILL INPUT THE FOLLOWING
# Team information: 
## - Player Names, with confirmation to get the right id, also the purchase price
## - itb
## - ft
## - when each chip was played
## - points each gw (optional)
# File Preferences
## - Name of your folder
## - Password if you'd like

from Agent import get_current_gw
from private_versions.constants import DROPBOX_PATH, STRING_SEASON, DEFAULT_REFERENCE_USER
from general_helpers import safe_make_folder, safe_read_csv, safe_to_csv
from private_versions.DefaultPersonalities import DEFAULT_PERSONALITY_NAMES, create_default_personality
import os
import shutil
import pandas as pd
from Accountant import make_name_df_full
import json

# query user input
def q(prompt):
    print(prompt, end='> ')
    return input()

# like range, in that end isn't used
def query_number(start, end, querystring='Type a #'):
    while True:
        try:
            x = q(querystring)
            if int(x) == float(x):
                x = int(x)
                if x >= start and x < end:
                    break
        except: 
            pass
        print("a number in the range please. decimals also aren't in the spirit")
    return x

def query_float(querystring, low = -float('inf'), high = float('inf')):
    while True:
        try:
            f = float(q(querystring))
            if f >= low and f <= high:
                return f
            else:
                print('number out of range')
        except:
            print('Not a number. Try again')


def is_yes(resp):
    return len(resp) > 0 and resp[0] in ('y', 'Y')

def get_teamname():
    while True:
        teamname = q('Team Name')
        if '.' in teamname:
            print('stop trying to hack')
        else:
            return teamname

def get_player_name_loop(name_df, name_list, name_list_lower):
    while True:
        word = q('\nPlayer Name').lower()
        print(word)
        options = [name_list[i] for (i, name) in enumerate(name_list_lower) if word in name]
        print('0) Search Again')
        for i, option in enumerate(options):
            print(f'{i+1}) {option}')
        x = query_number(0, len(options)+1, 'Which Number')
        if x == 0:
            continue
        else:
            chosen_player = options[x-1]
            print(f'SELECTED {chosen_player}')
            id = name_df.loc[name_df['name']==chosen_player]['id'].to_numpy()[0]
            return id


# LANDING SCREEN WELCOME AND ASSIGN TO HANDLER
while True:
    print('\n\nNOTE: Press ctrl+\\ at any time to stop execution')
    print('-----------------------------------------')
    print('|                                       |')
    print('          ATHENA WELCOMES YOU')
    print('    WHAT CAN I HELP YOU WITH TODAY ??\n')
    print('     0) END              ')
    print('     1) Create New Team') # biggest one, gets it all set up
    print('     2) Update Existing Team') # change the input files, or even the made moves
    print('     3) Remove Existing Team') # deletes the folder
    print('     4) Run Team') # runs overseer
    print('     5) Check Outputs')  # send message if haven't run yet, if so give them the command to open the excel (for made_moves and human_outputs)
    print('     6) General Statistics') # Prints some interesting lists
    print('|                                       |')
    print('-----------------------------------------')
    x = query_number(0, 7)

    if x == 0:
        print('Thanks for hanging out! :)')
        break

    elif x == 1:
        print('Creating a NEW team\n')
        #gw = get_current_gw()
        gw = query_number(1, 39, 'what gameweek is it')
        

        ##### username and password
        teamname = q('What should we name your team')
        team_folder = DROPBOX_PATH + 'ClientPersonalities/' + teamname + '/'
        if os.path.exists(team_folder):
            print('Team already exists. If you would like to create a new team with this name, delete the old team')
            exit()
        resp = q('Would you like to make a password so only you can touch this team?\n')
        if is_yes(resp):
            password = q('Choose a Password')

        ##### selecting personality and creating folder
        print_options = DEFAULT_PERSONALITY_NAMES + ['Design your own (Time Est. 3 Mins)']
        print('\nWhat personality do you want. You can use a preset or partially create your own.')
        for i, option in enumerate(print_options):
            print(f'{i+1}) {option}')
        x = query_number(1, 4)
        if x != len(print_options): #preset

            # FOLDER
            reference_user = print_options[x - 1]
            reference_folder = DROPBOX_PATH + reference_user + '/'
            if not os.path.exists(reference_folder):
                reference_folder = DROPBOX_PATH + DEFAULT_REFERENCE_USER + '/'
            # PERSONALITY
            personality = create_default_personality(STRING_SEASON, team_folder, reference_user)

        else: # create your own
            # FOLDER 
            reference_folder = DROPBOX_PATH + DEFAULT_REFERENCE_USER + '/'
            # PERSONALITY
            personality = create_default_personality(STRING_SEASON, team_folder, DEFAULT_REFERENCE_USER)
            
            print('\nCREATING NEW PERSONALITY. QUESTIONAIRRE')
            print('-----------------------------------------------\n')
            if is_yes(q('Would you be willing to transfer in a player with .75 chance of playing?')):
                personality['allowed_healths'] = ['a','d']
            
            print('\nNow you can enter players you would never want to own')
            print('Type in most of there name, the player closest to this will be matched')
            print('Simply press enter without typing at any point to end the blacklist')
            print('-----------------------------------------------')
            bad_players = []
            while True:
                player = q('')
                if player == '' or player == '\n':
                    personality['bad_players'] = bad_players
                    break 
                bad_players.append(player)

            print('\nSelect how much you would like to weight bench point relative to field points.')
            print('For exapmle, the default (as of Feb 2023) uses 0.15 and 0.015')
            print('This means that when calculating the score of a team, it is valued at')
            print('1*Score Starters + .15 * Score Bench on regular gameweeks, and')
            print('1*Score Starters + .015 * Score Bench on freehit gameweeks')
            print('-----------------------------------------------')
            a = query_float('Regular gw Bench Factor')
            b = query_float('Freehit gw Bench Factor')
            personality['bench_factors'] = (a, b)

            print('\nNow you must select what factors influence your transfer.')
            print('Select weights for value, next_match, and next_six matches')
            print('We will scale your numbers to sum to 1')
            print('Players are scored in the 3 categories, and their total dictates their overall worth of being transferred in')
            print('-----------------------------------------------')
            a = query_float('Score for value (projected pts / price)', low = 0.0001)
            b = query_float('Score for projected pts next gw', low = 0.0001)
            c = query_float('Score for projected pts next 6 gwks', low = 0.0001)
            personality['value_vector'] = [x / (a + b + c) for x in [a,b,c]] # scale to 1
            

            print('\nWhen is the earliest week you would want to play your chips?')
            print('-----------------------------------------------')
            personality['freehit'] = query_number(2, 39, 'Freehit')
            personality['triple_captain'] = query_number(2, 39, 'Triple Captain')
            personality['bench_boost'] = query_number(2, 39, 'Bench Boost')
            wca = query_number(2, 39, 'First Wildcard')
            wcb = query_number(2, 39, 'Second Wildcard')
            personality['wildcard'] = [wca, wcb]

            print('\nWhat is the max hit you would ever take, if a single hit is 4?')
            print('Support for 0, 4, 8, 12 (optimizing 6 transfers takes 80 minutes)')
            print('-----------------------------------------------')
            maxhit = 1
            while maxhit % 4 != 0:
                maxhit = query_number(0, 13, 'Max Hit')
            personality['max_hit'] = maxhit 

            
            print('\nLast thing, we must construct your Minimum Requirements and Hesitancy for making transfers?')
            print('For each possible situation, you will need to enter two numbers')
            print('\nMin Delta means the lowest increase in expected 6 week points you would be willing to make a transfer for')
            print(' -- Some example numbers for if you have 2 transfers is 1:0, 2: .75, 3: 1.5, 4: 2.25')
            print('\nHesitancy means how hesitant you are to make a given transfer')
            print('Numbers from 0 to 1 are allowed, but realistically stay between 0.25 and 0.75')
            print('anything <0.25 will almost always be chosen and more than 0.75 will almost never be')
            print(' -- Example hesitancy numbers for 2 ft = 1:0.4, 2:0.3, 3:0.6, 4:0.7')
            print('We see that person prefers to make 2 transfers when having 2 ft, then 1, and much less common 3 and 4')
            print('-----------------------------------------------')
            print('This section is a bit esoteric, would you like to skip it')
            if query_number(0, 2, '1 to continue, 0 to skip'):
                print('-----------------------------------------------')
                extra_transfers = int(maxhit / 4)
                hes = {1: {key: 0 for key in range(0, 2+extra_transfers)}, 
                    2: {key: 0 for key in range(1, 3+extra_transfers)}}
                delt =  {1: {key: 0 for key in range(0, 2+extra_transfers)}, 
                    2: {key: 0 for key in range(0, 3+extra_transfers)}}
                for num_ft in (1,2):
                    for num_transfers in range(0, num_ft + extra_transfers + 1):
                        print(f'\n {num_ft} Free Transfers and making {num_transfers} transfers')
                        delt[num_ft][num_transfers] = query_float('Delta', 0, 100)
                        if num_ft == 2 and num_transfers == 0:
                            print('hesitancy does not apply here, some transfer will be made')
                        else:
                            hes[num_ft][num_transfers] = query_float('Hesitancy', 0, .99999)

            
        # executing on the folder
        shutil.copytree(reference_folder, team_folder)

        ##### has moved?, itb, ft, chips
        print(' ------------------------------------------------------')
        print(' |                                                    |')
        print('   Now I need your team status to finalize the setup ')
        print(' |                                                    |')
        print(' ------------------------------------------------------')
        moved = is_yes(q('Have you made your moves already this week? (y or n)'))
        print('How many Free Transfers')
        if moved:
            ft = query_number(-1, 3)
            ft += 1
        else:
            ft = query_number(0, 3)
        itb = q('How much in the bank? Enter as an integer (e.g. for 1.3 million, enter 13)')
        fh = q('What week did you play freehit? Enter 38 for unplayed')
        bb = q('What week did you play bench boost? Enter 38 for unplayed')
        tc = q('What week did you play triple captain? Enter 38 for unplayed')
        wc = q('What week did you play the current wildcard?\nIf unplayed, enter the last gameweek to play THIS CURRENT wildcard (either 38 or something around 18)\n')

        ##### optional points each gw
        print('\n\n$$$ OPTIONAL, TYPE IN YOUR POINTS FOR EACH GAMEWEEK $$$\n$$$ This is used to make decisions on wildcarding $$$')
        print(' ------------------------------------------------------')
        print('1) Enter all points')
        print('2) Enter just last 6 gwks')
        print('3) Use default points')
        x = query_number(0, 4)
        gw_scores = [None] * 38
        reference_scores = safe_read_csv(reference_folder + 'human_inputs_meta.csv')
        if x == 1:
            for iter_gw in range(1, gw):
                gw_scores[iter_gw - 1] = query_number(0, 200, querystring=f'Score gw {iter_gw}')
        elif x == 2:
            for iter_gw in range(1, gw-6):
                gw_scores[iter_gw - 1] = reference_scores.loc[0, f'points_gw_{iter_gw}']
            for iter_gw in range(gw-6, gw):
                gw_scores[iter_gw - 1] = query_number(0, 200, querystring=f'Score gw {iter_gw}')
        elif x == 3:
            for iter_gw in range(1, gw):
                gw_scores[iter_gw - 1] = reference_scores.loc[0, f'points_gw_{iter_gw}']

        ##### player names with purchase price
        print("\n\n^!^ LAST SECTION --------- THE PLAYERS ^!^ ")
        print("^!^ Again, prices are INTEGERS!! (10x) ^!^ ")
        print(' ------------------------------------------------------')
        players = []
        name_df = make_name_df_full()
        name_list = name_df['name'].to_numpy()
        name_list_lower = [word.lower() for word in name_list]
        for n in range(15):
            id = get_player_name_loop(name_df, name_list, name_list_lower)
            price = query_number(0, 200, querystring='Purchase Price')
            players.append([id, price])

        ##### Write everything to the folders (including a personality.txt file), and overwriting made_moves
        meta_df_data = [[itb, ft, wc, bb, tc, fh] + gw_scores]
        meta_df_columns = ['itb', 'ft', 'wc', 'bb', 'tc', 'fh'] + [f'points_gw_{gw}' for gw in range(1, 39)]
        meta_df = pd.DataFrame(meta_df_data, columns=meta_df_columns)
        meta_df.to_csv(team_folder + 'human_inputs_meta.csv')

        players_data = players
        players_columns = ['player', 'purchase_value']
        players = pd.DataFrame(players_data, columns=players_columns)
        players.to_csv(team_folder + 'human_inputs_players.csv')

        os.remove(team_folder + 'human_outputs.csv')

        df = pd.read_csv(team_folder + 'made_moves.csv')
        df = df.loc[df['gw']<0, :] # we don't want any of this info, it is all for the consumer
        if moved:
            new_row = [[gw, 0,0,0,0,0,'normal']]
            df = pd.concat([df, new_row], axis=1)
        df.to_csv(team_folder + 'made_moves.csv')

        with open(team_folder + "personality.json", "w") as outfile:
            json.dump(personality, outfile)

    elif x == 2:
        print('Oh did you make a different transfer than you were suggested to make?')
        print('Here you can make changes to your files')
        print('This is not implemented yet because you may as well just make a new team')
        print('Please delete your old team and make a new one. Thanks')

        # realistically just need to get itb and ft
        # and then allow them to do player substitutions
        ## show the team, and then say would you like to do transfers

    elif x == 3: 
        teamname = get_teamname()
        try: 
            team_folder = DROPBOX_PATH + 'ClientPersonalities/' + teamname + '/'
            shutil.rmtree(team_folder)
            print(f'{teamname} has been successfully removed')
        except:
            print(f'{teamname} could not be found')

    elif x == 4:
        team_folder = DROPBOX_PATH + 'ClientPersonalities/' + get_teamname() + '/'
        from Overseer import FPL_AI
        personality = json.load(open(team_folder + "personality.json"))
        # json auto changes the inner dict keys to be strings rather than integers, so we have to change back
        for key in ['hesitancy_dict', 'min_delta_dict']:
            print('in the iteration')
            personality[key] = {int(ok):{int(k):v for k,v in inner_dict.items()} \
                 for ok,inner_dict in personality[key].items()}
        print(personality['hesitancy_dict'])
                
        ai = FPL_AI(**personality)
        print('BE PATIENT -- RUNNING TEAM -- \nWILL TAKE EITHER <5 MINUTES OR UP TO AN HOUR, \ndepending on if other teams have been run today')
        ai.make_moves()

    elif x == 5:
        print('check outputs needs to be implemented')


    elif x == 6:
        print('Week statistics Yet to be Implemented')
