''' 3 sections
General Helpers
Reused functions from the old (inefficient) search
The New Search Functions
'''
import pandas as pd 
import numpy as np 
from itertools import combinations
from collections import Counter
from constants import DROPBOX_PATH, VERBOSITY
import time
import random
from general_helpers import get_columns_containing, safer_eval




############################### ---------- ####################################
############################# GENERAL HELPERS #################################
############################### __________ ####################################

# chooses top 2 players that have health > 75% 
# currently updating to make not same team captain
def find_healthy_best_captains(starter_df, health_df):
    starter_df = starter_df.sort_values('expected_pts_N1',ascending=False).reset_index(drop=True)
    healthy = health_df.loc[health_df['status'].isin(['a','i'])]['element']
    healthy_df = starter_df.loc[starter_df['element'].isin(healthy)]

    good_starters = healthy_df.shape[0]
    if good_starters == 0: # we just go with the initial best
        captain = starter_df.iloc[0,:]['element']
        captain_pts = starter_df.loc[starter_df['element']==captain]['expected_pts_N1'].tolist()[0]
        vice_captain = starter_df.iloc[1,:]['element']
    elif good_starters == 1: # best healthy, and second best 
        captain = healthy_df.iloc[0,:]['element']
        captain_pts = starter_df.loc[starter_df['element']==captain]['expected_pts_N1'].tolist()[0]
        vice_captain = starter_df.loc[starter_df['element']!=captain].iloc[0,:]['element'] #best not the captain
    else: #get two healthy starters from 
        captain = healthy_df.iloc[0,:]['element']
        captain_pts = starter_df.loc[starter_df['element']==captain]['expected_pts_N1'].tolist()[0]
        vice_captain = healthy_df.iloc[1,:]['element']
        
    return captain, vice_captain, captain_pts
    

# prints the choices it was thinking of doing the screen
def transfer_option_names(scoreboard, name_df, num_transfers=0):
    print(scoreboard)
    for index, row in scoreboard.iterrows():
        players_in = []
        players_out = []
        for player_in in safer_eval(row['inbound']):
            players_in.append(name_df.loc[name_df['element']==player_in]['name'].tolist()[0])
        for player_out in safer_eval(row['outbound']):
            players_out.append(name_df.loc[name_df['element']==player_out]['name'].tolist()[0])
        if VERBOSITY['brain_important']: 
            print('Option Number ', index + 1, ' ~~~ ', 'players in= ', players_in, 'players out= ', players_out)

# renames a column for reuse of other functions
def rename_expected_pts(df):
    target_column = get_columns_containing(['expect'], df).columns[0]
    return df.rename(columns={target_column: 'expected_pts'})

# gets rid of injured players from the scoreboard
def filter_transfering_healthy_players(scoreboard, team_players, allowed_healths):
    scoreboard = scoreboard.reset_index(drop=True)
    injured_on_my_team = team_players.loc[~team_players['status'].isin(allowed_healths)]['element'].tolist()

    injured_dict = {}
    for index, row in scoreboard.iterrows():
        injured = 0
        for player in safer_eval(row['outbound']):
            if player in injured_on_my_team:
                injured += 1
        injured_dict[index] = injured 
    top_injured = max(injured_dict.values())
    valid_transfer_indices = [x for x in injured_dict if injured_dict[x] == top_injured]
    scoreboard = scoreboard.iloc[valid_transfer_indices, :]
    return scoreboard

# order the field player bench
# @return: list descending
def get_bench_order(bench_df):
    no_keepers = bench_df.loc[bench_df['position']!=1]
    sorted_bench = no_keepers.sort_values('expected_pts', ascending=False)
    return sorted_bench['element'].tolist()

# order the entire bench with keeper at position 0
# @return: list descending
def get_bench_order_with_keeper(bench_df):
    no_keepers = bench_df.loc[bench_df['position']!=1]
    sorted_bench = no_keepers.sort_values('expected_pts', ascending=False)
    field_bench = sorted_bench['element'].tolist()
    keeper = bench_df.loc[~bench_df['element'].isin(field_bench)]['element'].to_list()
    return keeper + field_bench

# random shuffles on the wildcard to encourage proper convergence
def randomly_shuffle_n_players(team_players, transfer_market, sell_value, bench_factor, n, how_often):
    if random.randint(0, how_often) != 1:
        return team_players
    
    # get rid of n of the team players, replace with random n from transfer market
    outbound = set(random.sample(team_players['element'].to_list(),n))
    positions = team_players.loc[team_players['element'].isin(outbound)]['position'].to_list()
    
    new_players = []
    for pos in positions:
        options = transfer_market.loc[(~transfer_market['element'].isin(team_players['element'])) & (transfer_market['position']==pos) & (~transfer_market['element'].isin(new_players))]['element'].to_list()
        choice = random.choice(options)
        new_players.append(choice)
    inbound = transfer_market.loc[transfer_market['element'].isin(new_players)]
        
    old_players = team_players.loc[~team_players['element'].isin(outbound)]
    new_team = pd.concat([old_players, inbound], axis=0)
    return new_team


###################### -------------------------------- #######################
###################### REUSED FUNCTIONS FROM OLD SEARCH #######################
###################### ________________________________ #######################

# @param: df, set, df, float, float
# ----- df's have df['status', 'element','position','team','value','expected_pts']
# @return: False if unfeasible, else returns the delta value
def check_feasibility_and_get_delta(inbound, outbound, team_players, sell_value, bench_factor):
    ''' create what new team would be '''
    old_players = team_players.loc[~team_players['element'].isin(outbound)]
    new_team = pd.concat([old_players, inbound],axis=0, sort=True)

    ''' team-3 check '''
    if max(new_team['team'].value_counts()) > 3:
        return False

    ''' team value check ''' 
    if new_team['value'].sum() > sell_value:
        return False

    ''' team_players score doesn't need to be sorted '''
    current_score = get_points(team_players, bench_factor)
    new_score = get_points(new_team, bench_factor)
    return new_score - current_score


# For each position, removes players who have greater than max_better
# players above them in expected points with <= price.
# with proper indexing
# @param: transfer_market - df['status', 'element','position','team','value','expected_pts']
#         bad_players := list-like containing elements of players already on my_team
#         allowed_healths := list of char, only players with this status can stay
#         max_better :=  integer
# @return: sorted transfer list without irrelevants
def kill_irrelevants_and_sort(transfer_market, bad_players, allowed_healths, max_better):
    
    transfer_market = transfer_market.loc[~transfer_market['element'].isin(bad_players)]
    transfer_market = transfer_market.loc[transfer_market['status'].isin(allowed_healths)]

    total_relevant = []
    for position in transfer_market['position'].unique():
        df = transfer_market.loc[transfer_market['position']==position]
        df = df.sort_values(['value', 'expected_pts'], ascending=[True, False]) #sort by low price, tiebreak on higher pts going first to prevent tons of 4.0

        top_scores = [-1] * (max_better+1) #top score on the left
        for row in df.iterrows():
            row = row[1]
            score = row['expected_pts']
            if score > top_scores[-1]:
                top_scores[-1] = score #replace last place score
                top_scores.sort(reverse=True) #list remains descending
                total_relevant.append(row)

    df_total = pd.concat(total_relevant, axis=1).T
    df_total = df_total.sort_values('expected_pts',ascending=False).reset_index(drop=True)
    return df_total 

#helper for get points but also for picking team in brain.py
#@param: typical team_players df
#@return: list of starters, list of bench players 
def get_starters_and_bench(team_players):
    team_players = team_players.sort_values('expected_pts',ascending=False).reset_index(drop=True)
    keepers = team_players.loc[team_players['position']==1]['element'].tolist()
    full = [ keepers[0] ]
    bench = [ keepers[1] ]
    used_elements = keepers

    top_three_defenders = team_players.loc[team_players['position']==2].iloc[0:3,:]['element'].tolist()
    full = full + top_three_defenders
    used_elements= used_elements + top_three_defenders

    top_forward = team_players.loc[team_players['position']==4]['element'].tolist()[0]
    full.append(top_forward)
    used_elements.append(top_forward)


    remaining = team_players.loc[~team_players['element'].isin(used_elements)]['element'].to_list()
    full = full + remaining[:6]
    bench = bench + remaining[-3:]
    return full, bench


# go from the team to the points 
# team players doesn't need to be sorted, but needs to have expected_pts column 
def get_points(team_players, bench_factor):
    team_players = rename_expected_pts(team_players)
    starters, benchwarmers = get_starters_and_bench(team_players)

    starter_scores = team_players.loc[team_players['element'].isin(starters)]['expected_pts']
    bench_scores = team_players.loc[team_players['element'].isin(benchwarmers)]['expected_pts']
    return starter_scores.sum() + bench_factor * bench_scores.sum()
     


############################### ---------- ####################################
############################### NEW SEARCH ####################################
############################### __________ ####################################
''' Search Summary 
1) Set up a bunch of datastructures
2) Carefully loop through all combinations of players greedily
    a) update datastructures as we go
        I) these allow us to cut out early
    b) pass information down to reuse computations
3) Terminate when certain we have the top n combos
'''

''' ############### SETTING UP THE DATASTRUCTURES ################ ''' 
def tm_search_order(transfer_market):
    transfer_market = transfer_market.sort_values('expected_pts', ascending=False).reset_index(drop=True)
    order = {}
    for i, row in transfer_market.iterrows():
        position, element = row[['position', 'element']]
        order[i] = (position, element)
    return order, len(order)

def team_outbound_order(squad):
    out_order = {}
    for i in squad:
        out_order[i] = squad[i]['element'].to_list()
    return out_order

def make_positional_dicts(transfer_market, full_team_players, starters, bench_factor):
    team_players = full_team_players.copy()
    team_players.loc[~team_players['element'].isin(starters), 'expected_pts'] = team_players['expected_pts'] * bench_factor
    transfer_market = transfer_market.sort_values('expected_pts', ascending=False)
    team_players = team_players.sort_values('expected_pts', ascending=True)
    tm = {i:transfer_market.loc[transfer_market['position']==float(i)] for i in list(range(1, 5))}
    squad = {i:team_players.loc[team_players['position']==float(i)] for i in list(range(1, 5))}
    return tm, squad

# gives value info, team, and delta points functions for each possible 1x1 transfer
def make_big_transfer_dict(tm_dict, team_dict):
    big_dict = {}
    for i in list(range(1,5)):
        tm, squad = tm_dict[i], team_dict[i]
        for outbound in squad['element']:
            out_price, out_team = squad.loc[squad['element']==outbound][['value', 'team']].T.squeeze()
            for inbound in tm['element']:
                in_price, in_team, in_points = tm.loc[tm['element']==inbound][['value', 'team', 'expected_pts']].T.squeeze()
                big_dict[(outbound, inbound)] = {
                    'cost': in_price - out_price,
                    'teams': (out_team, in_team),
                    'points': in_points
                }
    return big_dict


def make_max_price_swings(tm, squad):
    price_swings = []
    for pos in range(1,5):
        min_cost_pos = tm[float(pos)]['value'].min()
        for out in squad[float(pos)]['value']:
            price_swings.append(out - min_cost_pos)
    price_swings.sort(reverse=True)
    n_swings = {0:0}
    total = 0
    for n in range(1,16):
        total += price_swings[n-1]
        n_swings[n] = total 
    return n_swings


''' ############### HELPERS FOR SCORING TRANSFERS ################ ''' 
def update_team_vector(team_vector, in_team, out_team):
    change = np.zeros(21)
    change[int(out_team)] = -1 
    change[int(in_team)] = 1
    return team_vector + change

# sorted points by position and last is greatest
# key that maps the element to (position, index_in_that_dict)
def get_pos_ordered_pts(full_team_players):
    #preprocessing will be unnecessary if this works get rid of previous adding bench scores
    team_players = full_team_players.sort_values('expected_pts', ascending=True)
    squad = {i:team_players.loc[team_players['position']==float(i)] for i in list(range(1, 5))}
    
    op_dict = {}
    op_key = {}
    for pos in squad:
        op_dict[pos] = squad[pos]['expected_pts'].to_list()
        elements = squad[pos]['element'].to_list()
        new_dict = {elements[i]:(pos, i) for i in range(len(elements))}
        op_key.update(new_dict)
    return op_dict, op_key
    


''' ############### HELPERS FOR SCORING TRANSFERS ################ ''' 

# should be sorted because point_deltas goes in ascending for teamout but descending for transin
def score_transfers_initialization(op_key, point_deltas, transfer_list):
    forb_spots = {1:set(),2:set(),3:set(),4:set()}
    inb_pts = {1:[],2:[],3:[],4:[]}
    for trans, pts in zip(reversed(transfer_list), reversed(point_deltas)): 
        out = trans[0]
        pos, i = op_key[out] 
        forb_spots[pos].add(i)

        inb_pts[pos].append(pts)
        
    return forb_spots, inb_pts

# return remaining a,b, and the points 
def top_n_in_sorted_lists(a,b,n):
    total = 0
    for iteration in range(n):
        if len(a) == 0:
            remaining = n - iteration
            return b[:-remaining], total + sum(b[-remaining:])
        if len(b) == 0:
            remaining = n - iteration
            return a[:-remaining], total + sum(a[-remaining:])

        left_higher = a[-1] > b[-1]
        if left_higher:
            total += a.pop()
        else:
            total += b.pop()
    return a+b,total 

''' ################# SCORING THE TRANSFERS ###############''' 
def score_transfers_v2(pos_ordered_pts, op_key, transfer_list, point_deltas, bench_factor, first_n_dict):
    forbidden_spots, inb_ordered_pts = score_transfers_initialization(op_key,point_deltas, transfer_list)
    remaining_team_players = {pos: [v for i,v in enumerate(l) if i not in forbidden_spots[pos]] for pos,l in pos_ordered_pts.items()}
    total_pts = 0
    rest = []
    for pos in range(1,5):
        num = first_n_dict[pos]
        leftovers, pts = top_n_in_sorted_lists(remaining_team_players[pos],inb_ordered_pts[pos], num)
        total_pts += pts 
        if pos==1:
            total_pts += leftovers[0] * bench_factor
        else: 
            rest += leftovers
    rest.sort()
    total_pts += sum(rest) - sum(rest[:3]) * (1-bench_factor)
    return total_pts 

''' ################# MAIN SEARCH FUNCTION ###############''' 
# bottleneck is computing the starters 
# 3 transfers in 10s, 4 transfers in (19s - 5 minutes), 5in 5min, 6 in <1.5hours 
    # the ranges are if it is obvious to if there are no good transfers
def search_v2(transfer_market, team_players, sell_value, num_transfers, num_options, bench_factor, protected_players):
    starters, bench = get_starters_and_bench(team_players)
    # trans and teams as positional dicts 
    tm, squad = make_positional_dicts(transfer_market, team_players, starters, bench_factor)
    root_outbound_order = team_outbound_order(squad)
    pos_ordered_pts, op_key = get_pos_ordered_pts(team_players)
    first_n_dict = {1:1, 2:3,3:0,4:1}
    
    # TO BE USED AS GLOBALS IN THIS RECURSIVE FUNCTION
    search_order, market_size = tm_search_order(transfer_market)
    start = time.time()
    trade_dict = make_big_transfer_dict(tm, squad)
    #position_size = {1: 2, 2: 5, 3: 5, 4:3}
    initial_team_vector = np.bincount(np.array([int(x) for x in team_players['team'].to_list()]))
    initial_team_vector.resize(21)
    max_price_swing = make_max_price_swings(tm, squad)
    money_itb = sell_value - team_players['value'].sum() 
    sorted_team_points = sorted(team_players['expected_pts'].to_list(), reverse=True)
    bench_total = team_players.loc[team_players['element'].isin(bench)]['expected_pts'].sum()*bench_factor
    base_score = get_points(team_players, bench_factor)

    TOTAL_PRICE_FAILS = 0
    TOTAL_TEAM_FAILS = 0
    TOTAL_HEURISTIC_FAILS = 0
    TOTAL_HEURISTIC_CHECKS = 0
    TOTAL_SCORE_CHECKS = 0
    TOTAL_POS_FAIL_IMMEDIATELY = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}

    def search_rec(scoreboard, outbound_order, transfer_list, n, start_index,\
            cost, team_vector, point_deltas, quick_fail):
        
        nonlocal TOTAL_PRICE_FAILS
        nonlocal TOTAL_TEAM_FAILS
        nonlocal TOTAL_HEURISTIC_FAILS
        nonlocal TOTAL_HEURISTIC_CHECKS
        nonlocal TOTAL_SCORE_CHECKS

        if start_index >= market_size: #error prevention case
            return scoreboard, True

        if n == 0: #base case
            min_delta = scoreboard['delta'].iloc[-1]
            fail_heuristic = False
            """RETHINK"""
            # Get the score 
            # If worse than min_delta, fail_heuristic, and in above check that index==0 to pass up, break either way
            TOTAL_SCORE_CHECKS += 1 #################
            delta = score_transfers_v2(pos_ordered_pts, op_key, transfer_list, point_deltas, bench_factor, first_n_dict) - base_score
            if delta < min_delta:
                fail_heuristic=True
            elif not quick_fail:
                inb = set([x[1] for x in transfer_list])
                outb = set([x[0] for x in transfer_list])
                scoreboard.iloc[-1,:] = str(inb), str(outb), delta # should just replace last row
                #print("New guy on scoreboard: ", str(inb), str(outb), delta)
                scoreboard = scoreboard.sort_values('delta',ascending=False).reset_index(drop=True)
                #scoreboard.sort_values('delta',ascending=False, inplace=True).reset_index(drop=True)
            return scoreboard, fail_heuristic


        # Recursion Steps
        at_least_one_success = False
        good_positions = {1,2,3,4}
        first_each_position = {}
        checked_heuristic_trigger_elements = set()
        for i in range(start_index, market_size):
            pos, element = search_order[i]
            if pos not in first_each_position:
                first_each_position[int(pos)] = (i, element)
                potential_heuristic_trigger = True
            else:
                potential_heuristic_trigger = False
            if pos not in good_positions:
                if good_positions == set():
                    break 
                continue 


            for index, out_player in enumerate(outbound_order[pos]):
                this_transfer = (out_player, element)

                info = trade_dict[this_transfer]
                this_cost = info['cost']
                out_team, in_team = info['teams']
                this_team_vector = update_team_vector(team_vector, in_team, out_team)
                this_point_deltas = point_deltas + [info['points']]
                quick_fail = False
                fail_heuristic = False
                ## INTERMEDIATE CHECKS FOR SPEED ##
                if cost + this_cost - max_price_swing[n-1] > money_itb:
                    TOTAL_PRICE_FAILS += 1##############
                    quick_fail = True 
                elif max(this_team_vector) - n > 2: #even if focus transfers on getting rid of team can't
                    TOTAL_TEAM_FAILS += 1 ############
                    quick_fail = True 


                ## DO WE RECURSE ##
                if not quick_fail or not at_least_one_success: #recurse necessary to check for fail_immediately
                    new_outbound_order = outbound_order.copy()
                    new_outbound_order[pos] = [[] if index == len(outbound_order[pos]) - 1 else outbound_order[pos][index+1:]][0]

                    scoreboard, fail_heuristic = search_rec(scoreboard, new_outbound_order,\
                        transfer_list + [this_transfer], n-1, i+1, cost+this_cost, this_team_vector,\
                        this_point_deltas, quick_fail=quick_fail)
                    if potential_heuristic_trigger and index == 0:
                        checked_heuristic_trigger_elements.add(element)

                    #don't have to check index==0 because only will call check_heuristic if so
                    if fail_heuristic: # this position is no longer possibly holding good people
                        TOTAL_POS_FAIL_IMMEDIATELY[n] += 1
                        if index==0:
                            #if n == num_transfers and pos in good_positions:
                            #    print("For ", num_transfers, " transfers, had terminating failure for position ", str(int(pos)), "at i=", i, "out of ", market_size-1)

                            good_positions.discard(pos)    
                            break            

                    else: #succeeded at least once
                        at_least_one_success = True
                        

        fail_heuristic = not at_least_one_success
        return scoreboard, fail_heuristic
    
    
    init_scoreboard = pd.DataFrame( [['set()','set()',0]] * num_options, columns= ['inbound','outbound','delta'] , dtype=object)
    scoreboard, _ = search_rec(init_scoreboard, root_outbound_order, [], num_transfers, 0, 0, initial_team_vector, [],False)
    #print("Price Fails: ", TOTAL_PRICE_FAILS, "Team Fails: ", TOTAL_TEAM_FAILS,\
    #    "Score Checks: ", TOTAL_SCORE_CHECKS, "Position Immediate Failures: ", TOTAL_POS_FAIL_IMMEDIATELY)
    return scoreboard