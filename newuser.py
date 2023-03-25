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

from Agent import get_current_gw
from constants import DROPBOX_PATH, STRING_SEASON, DEFAULT_REFERENCE_USER
from general_helpers import safe_make_folder, safe_read_csv, safe_to_csv, \
    get_user_folder_from_user, get_user_personality, save_user_personality
from DefaultPersonalities import DEFAULT_PERSONALITY_NAMES, create_default_personality
import os
print('here')
import shutil
import pandas as pd
from Accountant import make_name_df_full
print('there')
import json
from Overseer_helpers import add_to_orders, remove_from_orders

print('past imports')
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
        if '..' in teamname:
            print('stop trying to hack')
        else:
            if os.path.exists(get_user_folder_from_user('ClientPersonalities/' + teamname)):
                return teamname
            else:
                print("Not an existing TEAM")

def query_df_gw(filename):    
    df = safe_read_csv(filename)
    gw = query_number(0, 39, 'which gameweek you would like to look at, 0 for all')
    if gw != 0:
        df = df.loc[df['gw']==gw]
    return df  

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
            id = name_df.loc[name_df['name']==chosen_player]['element'].to_numpy()[0]
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
    print('     6) Activate Team (will run automatically like official athena teams)') # Prints some interesting lists
    print('     7) View Player Rankings') # Prints some interesting lists
    print('|                                       |')
    print('-----------------------------------------')
    x = query_number(0, 8)

    if x == 0:
        print('Thanks for hanging out! :)')
        break

    elif x == 1:
        print('Creating a NEW team\n')
        #gw = get_current_gw()
        gw = query_number(1, 39, 'what gameweek is it')
        

        ##### username and 
        teamname = q('What should we name your team')
        user = 'ClientPersonalities/' + teamname
        team_folder = get_user_folder_from_user(user)
        if os.path.exists(team_folder):
            print('Team already exists. If you would like to create a new team with this name, delete the old team')
            exit()

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
        itb = query_number(0, 1000, 'How much in the bank? Enter as an integer (e.g. for 1.3 million, enter 13)')
        fh = query_number(1, 39, 'What week did you play freehit? Enter 38 for unplayed')
        bb = query_number(1, 39,'What week did you play bench boost? Enter 38 for unplayed')
        tc = query_number(1, 39,'What week did you play triple captain? Enter 38 for unplayed')
        wc = query_number(1, 39,'What week did you play the current wildcard?\nIf unplayed, enter the last gameweek to play THIS CURRENT wildcard (either 38 or something around 18)\n')
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

        save_user_personality(user, personality)

    elif x == 2:
        print('Oh did you make a different transfer than you were suggested to make?')
        print('Here you can make changes to your files')

        team_folder = get_user_folder_from_user('ClientPersonalities/' + get_teamname())
        meta_inputs_file = team_folder + 'human_inputs_meta.csv'
        player_intputs_file = team_folder + 'human_inputs_players.csv'
        made_moves_file = team_folder + 'made_moves.csv'

        # ITB, FT, CHIPS
        print('First Need to Recalibrate the MetaData')
        df = safe_read_csv(meta_inputs_file)
        
        # same loop as x == 1
        gw = query_number(1, 39, 'what gameweek is it')
        moved = is_yes(q('Have you made your moves already this week? (y or n)'))
        print('How many Free Transfers')
        if moved:
            ft = query_number(-1, 3)
            ft += 1
        else:
            ft = query_number(0, 3)
        itb = query_number(0, 1000, 'How much in the bank? Enter as an integer (e.g. for 1.3 million, enter 13)')
        fh = query_number(1, 39, 'What week did you play freehit? Enter 38 for unplayed')
        bb = query_number(1, 39,'What week did you play bench boost? Enter 38 for unplayed')
        tc = query_number(1, 39,'What week did you play triple captain? Enter 38 for unplayed')
        wc = query_number(1, 39,'What week did you play the current wildcard?\nIf unplayed, enter the last gameweek to play THIS CURRENT wildcard (either 38 or something around 18)\n')
        df.loc[0, ['itb', 'ft', 'wc', 'bb', 'tc', 'fh']] = [itb, ft, wc, bb, tc, fh]
        safe_to_csv(df, meta_inputs_file)

        # edit the made moves 
        df = safe_read_csv(made_moves_file)
        df = df.loc[df['gw']<gw, :] # we don't want this gw
        if moved:
            new_row = pd.DataFrame([[gw, 0,0,0,0,0,'normal']], columns=df.columns)
            df = pd.concat([df, new_row], axis=0)
        df.to_csv(team_folder + 'made_moves.csv')
        
        # Change Player Loop
        ## show the team, and then say would you like to do transfers
        print('\nNow you can change what players you are listed as having.\nDo so at your own risk,\
            if you end up with the wrong number per position you will break things\nYou can also use this to adjust purchase price\n--------------------------')
        name_df = make_name_df_full()
        name_list = name_df['name'].to_numpy()
        name_list_lower = [word.lower() for word in name_list]
        df = safe_read_csv(player_intputs_file)
        while True:
            # print team 
            for i, row in df.iterrows():
                id, purchase_value = row 
                name = name_df.loc[name_df['element']==id]['name'].to_numpy()[0]
                print(f'{i}) {name}')

            # get someone to get rid of
            rid_num = query_number(0, 16, 'Pick a number to get rid of, 15 to be done')
            if rid_num == 15:
                break

            # get someone to bring in
            print('\nPlayer to bring in', end='')
            id = get_player_name_loop(name_df, name_list, name_list_lower)
            price = query_number(0, 200, querystring='Purchase Price')
            df.loc[rid_num, ['player', 'purchase_value']] = [id, price]


    elif x == 3: 
        teamname = get_teamname()
        try: 
            user = 'ClientPersonalities/' + get_teamname()
            # if in the running set, get rid of it
            remove_from_orders()

            # remove the folder
            team_folder = get_user_folder_from_user(user)
            shutil.rmtree(team_folder)
            print(f'{teamname} has been successfully removed')
        except:
            print(f'{teamname} could not be found')

    elif x == 4:
        personality = get_user_personality('ClientPersonalities/' + get_teamname()) 
        print('BE PATIENT -- RUNNING TEAM -- \nWILL TAKE EITHER <5 MINUTES OR UP TO AN HOUR, \ndepending on if other teams have been run today')
        from Overseer import FPL_AI             
        ai = FPL_AI(**personality)
        ai.make_moves()

    elif x == 5:
        team_folder = get_user_folder_from_user('ClientPersonalities/' + get_teamname())
        made_moves_file = team_folder + 'made_moves.csv'
        team_outputs_file = team_folder + 'human_outputs.csv'

        # Display Made Moves
        print('Showing when moves have been made')
        df = query_df_gw(made_moves_file)
        for i, row in df.iterrows():
            gw, yr, month, day, hour, num_transfers, chip = row.to_numpy()
            print(f'{month}/{day}/{str(yr)[2:]} , gw{gw} == {chip} - {num_transfers} transfer(s)')

        # Display Team Outputs
        print('\nShowing Current Chosen Team, \nwith projected ppg next game (N1) and next 6 (N6)\nYes, the bench is ordered')
        df = query_df_gw(team_outputs_file)
        for gw in df.copy()['gw'].unique():
            print(f'\nGAMEWEEK {gw}\n######################################')
            print('       NAME       |   N1   |   N6    |\n=======================================')
            dfgw = df.loc[df['gw']==gw].reset_index(drop=True)
            captain, vc, chip = dfgw.loc[0, ['captain', 'vc', 'chip']].to_numpy()
            for i in range(1, 16):
                player, score_one, score_six = dfgw.loc[:, f'name_{i}'].to_numpy()
                if player == captain:
                    player += ' (C)'
                if player == vc:
                    player += ' (vc)'
                player = player + ' '*max(0, (17 - len(player)))
                score_one = str(score_one) + ' '*(4-len(str(score_one)))
                score_six = str(score_six) + ' '*(4-len(str(score_six)))
                print(f'{player} |  {score_one}  |  {score_six}   |')
                if i == 11:
                    print ('--------------------------------------')
                    
            print('\n Chip is ', chip, '\n')


    elif x == 6:
        user = 'ClientPersonalities/' + get_teamname()
        add_to_orders(user)
        

    elif x == 7:
        full_df = safe_read_csv(DROPBOX_PATH + 'full_transfer_market.csv')
        name_df = make_name_df_full()
        full_df['name'] = full_df.apply(lambda row: name_df.loc[name_df['element']==row['element']]['name'].to_numpy()[0], axis=1)
        while True:
            print('\nPLAYER RANKINGS\n----------------\n0)EXIT\n1)Next Match Points\n2)Expected pts over 6 gwks\n----------------------')
            x = query_number(0, 3)
            if x > 0:
                col = ('expected_pts_full' if  x == 2 else 'expected_pts_N1')
                print('\n0) OVERALL RANKINGS\n1)goalkeepers\n2)defenders\n3)midfielders\n4)forwards\n---------------')
                x = query_number(0, 5)

                df = full_df.sort_values(by=col, ascending=False)
                if x != 0:
                    df = df.loc[df['position']==x]

                maxn = query_number(1, 10000, 'See only top how many values')

                # now displaying
                print(f'RANKING FOR {col}')
                for i, row in df.reset_index(drop=True).iterrows():
                    if i >= maxn:
                        break
                    player, score = row[['name', col]] 
                    playerstring = player + ' '*max(0, (32 - len(player)))
                    score = round(float(score), 2)
                    scorestring = str(score) + ' '*(4-len(str(score)))
                    rankstring = f'Rank {i + 1} ' 
                    rankstring += ' '*(9 - len(rankstring)) 
                    print(f'{rankstring} = {playerstring} |  {scorestring}  |')
            else:
                break

