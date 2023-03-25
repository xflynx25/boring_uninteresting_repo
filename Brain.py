################# SUMMARY #################
# Main Functions (--> means 'which calls'):
#  weekly_transfer --> choose_top_transfer_n --> top_transfer_options
#  free_hit_team/best_wildcard_team --> *the above*
#  pick_team
#  play_chips_or_no
#  figure_out_substitution
# Minor Helper Func
#  get_cheapest_team
# Overseer Support Func
#  match_positions
###########################################
import pandas as pd
from math import pi, tan, exp, factorial
from random import randint
from Brain_helpers import kill_irrelevants_and_sort, check_feasibility_and_get_delta,\
    get_points, get_starters_and_bench, find_healthy_best_captains, transfer_option_names, rename_expected_pts,\
    filter_transfering_healthy_players, get_bench_order, get_bench_order_with_keeper, randomly_shuffle_n_players,\
    search_v2, safer_eval
from constants import WILDCARD_DEPTH_PARAM, VERBOSITY
import time


# @param: transfer_market = df with status, element, position, team, value, expected_pts____ (either full or N1) !!need to be last
#               comes with team_players' prices adjusted to sell value
#         team_players = df with status, element, position, team, value(selling-value), 'expected_pts'
#         sell_value = int, maximum cost of all players 
#         num_transfers = int, how many transfers to make
#         num_options = int, how many transfer_options to return
# @return: pd dataframe with columns ((inbound), (outbound), delta)
def top_transfer_options(transfer_market, team_players, sell_value, num_transfers, num_options, bench_factor, player_protection, allowed_healths=['a']):
    transfer_market = rename_expected_pts(transfer_market)
    team_players = rename_expected_pts(team_players)
    # sort team players, make empty scoreboard df 
    team_players = team_players.sort_values('expected_pts',ascending=False).reset_index(drop=True)
    scoreboard = pd.DataFrame( [['set()','set()',0]] * num_options, columns= ['inbound','outbound','delta'] )

    '''We currently choose there to be n+1 options for same team redundancy, actually no check next line'''
    max_better = num_transfers - 1 # we recognize that this might not be ideal, but it speeds up our processing
    transfer_market = kill_irrelevants_and_sort(transfer_market, team_players['element'], allowed_healths, max_better)
    #print('sorted transfer_market top 25= ', transfer_market.iloc[:25, :])

    '''get the protected players that can't be transfered out from our team'''
    full_transfer_market = pd.concat([team_players, transfer_market], axis=0, ignore_index=True)
    sorted_market = full_transfer_market.sort_values('expected_pts',ascending=False).reset_index(drop=True)
    protected_players = sorted_market.iloc[:player_protection, :]['element'].tolist()\
        
    #####start = time.time()
    scoreboard = search_v2(transfer_market, team_players, sell_value, num_transfers,\
        num_options, bench_factor, protected_players)
    if VERBOSITY['brain']: 
        print(set(scoreboard['delta'].to_numpy()))
    #####print("New Search: ", round(time.time()-start, 8))
    #print(scoreboard)
    return scoreboard


# scoreboard : pd dataframe with columns list(set(inbound)), list(set(outbound)), delta), wrapping set in list so keep in df
# full_tm = df with (status, element, position, team, value, expected pts N1, expected ptsmax)
# team_players = "
# value_vector: 3 floats representing relative importance between (score/price), next game, max game deltas ... can also 
#   include fourth value which is a list containing allowed healths 
# min_delta: we throw out those that have bad results for the current gw as we will not consider them
# @return: series with index of inbound, outbound, delta, delta_N1, worth, ####ranking for those 3, and totalRanking
def choose_top_transfer_n(scoreboard, full_transfer_market, team_players, sell_value, bench_factor, value_vector, min_delta):
    def add_N1_and_worth_columns(row, full_transfer_market, team_players):
        #row = scoreboard_row.copy()
        inb_players = safer_eval(row['inbound'])
        inbound = full_transfer_market.loc[full_transfer_market['element'].isin(inb_players)].drop('expected_pts_full', axis=1)

        inbound = rename_expected_pts(inbound)
        outbound = safer_eval(row['outbound'])
        players = team_players.drop('expected_pts_full', axis=1)
        players = rename_expected_pts(players)

        delta = check_feasibility_and_get_delta(inbound, outbound, players, sell_value, bench_factor)

        '''that was the next game, now get the values or worths '''
        total_cost = inbound['value'].sum()
        worth = row['delta'] / total_cost

        delt_and_worth = pd.Series([delta, worth], index = ['delta_N1','worth'])
        return pd.concat([row, delt_and_worth], axis=0)

    def best_option(scoreboard, value_vector, ranking=True, top_scores=0):
        a,b,c = value_vector[:3]
        if ranking:
            scoreboard['totalRanking'] = a*scoreboard['worthRanking'] + b*scoreboard['delta_N1Ranking'] + c*scoreboard['deltaRanking']
            scoreboard.drop(['worthRanking', 'deltaRanking', 'delta_N1Ranking'], axis=1)
        else:
            worth_max, n1_max, full_max = top_scores
            scoreboard['totalRanking'] = a*scoreboard['worth']/worth_max + b*scoreboard['delta_N1']/n1_max + c*scoreboard['delta']/full_max
            scoreboard.drop(['worth', 'delta_N1', 'delta'], axis=1)

        winner = scoreboard.sort_values('totalRanking',ascending=False).reset_index(drop=True).iloc[0,:] #best has highest total ranking
        #print(scoreboard)
        return winner.drop(['totalRanking'])

    '''clean up the table for only relevant'''
    #print("Full transfermarket: ", full_transfer_market)
    all_playas = []
    for x in scoreboard['outbound'].to_list() + scoreboard['inbound'].to_list():
        all_playas += [i for i in safer_eval(x)]
        #for playa in set(all_playas):
            #print(f"Player {playa} = full:-: {full_transfer_market.loc[full_transfer_market['element']==playa]['expected_pts_full'].to_numpy()} anand next:-:{full_transfer_market.loc[full_transfer_market['element']==playa]['expected_pts_N1'].to_numpy()}")
    #print('og scoreboard\n', scoreboard)


    scoreboard = scoreboard.loc[scoreboard['delta']>0] #drop extra rows in the case of not enough satisfiable transfers
    if scoreboard.shape[0] == 0: #already perfect team
        print('already perfect team')
        return pd.Series()
    if len(value_vector) > 3: #means we want to get rid of our injured players
        scoreboard = filter_transfering_healthy_players(scoreboard, team_players, value_vector[3])
    FULL_SCOREBOARD = scoreboard.apply(lambda x: add_N1_and_worth_columns(x, full_transfer_market, team_players),axis=1, result_type='expand')
    if VERBOSITY['brain']: 
        print('full scoreboard: \n', FULL_SCOREBOARD)
    FULL_SCOREBOARD = FULL_SCOREBOARD.loc[FULL_SCOREBOARD['delta_N1'] >= min_delta] #drop those that we would reject on low N1
    if FULL_SCOREBOARD.shape[0] == 0: #already perfect team
        print('already perfect team considering Next1')
        return pd.Series()

    #develop ranking columns
    def get_top_score(val_type):
        full_scoreboard = FULL_SCOREBOARD.sort_values(val_type,ascending=True).reset_index(drop=True)
        full_scoreboard[f'{val_type}Ranking'] = full_scoreboard.index
        return float(full_scoreboard[val_type].max())

    top_delta = get_top_score('delta')
    top_N1 = get_top_score('delta_N1')
    worth = get_top_score('worth')
    #use value vector to choose the best, USE EITHER RANKING OR ABSOLUTE (relative to top scorer)
    top_scores = [worth, top_N1, top_delta]
    return best_option(FULL_SCOREBOARD, value_vector, ranking=False, top_scores= top_scores) #ABSOLUTE
    



# full_tm = df with (status, element, position, team, value, expected pts N1, expected pts full)
# team_players = "
# sell_value = float of selling price 
# free_transfers, max_hit = ints
# choice_factors = value_vector: 3 floats representing relative importance between value, next game, max game deltas
#       hesitancy_dict: 0 meaning always transfer, 1 meaning never
#           #this is nested, based on ft, then a dict of num_transfers
#       quality_factor: float, the exponent we will take the ratio of quality factor to, probably between 0 and 10
#       season_avg_delta_dict = avg delta on transfers of that length keys = 1,2 etc, vals=floats
#       num_options, bench_factor (help in selecting the transfer options and delta)
# @return: the selected transfer this week, in series form (inbound, outbound, delta, delta_N1, worth)
#           other return is list containing num_transfers (>=1) and corresponding delta 
#
# nOTE: we are still verifying that the N1 result beat the reqiured for the gw, even tho we currently
#   are throwing them out somewhere else as well. Since this might not always be the case. 
def weekly_transfer(full_transfer_market, team_players, sell_value, free_transfers, max_hit, choice_factors, player_protection,\
    allowed_healths=['a'], visualize_names=False, name_df = None):
    '''initializing all the parameters'''
    value_vector, hesitancy_dict, quality_factor, season_avg_delta_dict, min_delta_dict, num_options, bench_factor = choice_factors
    transfer_market = full_transfer_market.drop('expected_pts_N1', axis=1)
    search_team_players = team_players.drop('expected_pts_N1', axis=1)
    max_transfers = free_transfers + max_hit // 4
    #print('free transfers, maxtransfers', free_transfers, max_transfers)

    '''get best transfer for each num transfers allowed by personality'''
    choices = []
    for num_transfers in range(1, max_transfers + 1):
        #print('num options in here is ,', num_options)
        scoreboard = top_transfer_options(transfer_market, search_team_players, sell_value,\
            num_transfers, num_options, bench_factor, player_protection, allowed_healths)

        if visualize_names:
            visual_scoreboard = scoreboard.loc[scoreboard['delta']>0] 
            transfer_option_names(visual_scoreboard, name_df, num_transfers)

        top_option = choose_top_transfer_n(scoreboard, full_transfer_market, team_players,\
            sell_value, bench_factor, value_vector, min_delta_dict[free_transfers][num_transfers])
        if top_option.shape[0] > 0: #if there was an improvement found
            choices.append(top_option) #will be a series
    

    '''store choice deltas in df for logging purposes'''
    choices_list = []
    for choice in choices:
        choices_list.append((len(safer_eval(choice['outbound'])), choice['delta']))
    choice_report = pd.DataFrame(choices_list, columns=['num_transfers', 'delta'])

    save_ft = pd.Series(['set()','set()',0], index=['inbound', 'outbound', 'delta'])
    if choices == []: #No improvements on the team have been found
        return save_ft, choice_report


    '''Use tunable parameters to decide which number of transfers to make''' 

    def tickets(n, score, delta_N1score, free_transfers, hesitancy_dict, quality_factor, season_avg_delta, min_delta_dict):
        if n == 0 and free_transfers == 2: #don't save when already capped
            return 0
        if delta_N1score < min_delta_dict[free_transfers][n]: #if transfer doesn't meet our requirement to be good this week
            return 0

        hes = hesitancy_dict[free_transfers][n]
        hes = abs( (10000*hes - 1) / 10000 )
        hes_factor = exp(- tan(pi * (hes + 1/2))) 

        if not season_avg_delta or n == 0:
            quality = 1
        else:
            quality = score/season_avg_delta #will be False if no value sevaluated yet
        return round( 10000 * hes_factor * quality**quality_factor)

    def auction(choices, free_transfers, hesitancy_dict, quality_factor, season_avg_delta_dict, min_delta_dict):
        buckets = {
            0: tickets(0, 1, 0, free_transfers, hesitancy_dict, quality_factor,  1, min_delta_dict)
        }
        for choice in choices: 
            n = len(safer_eval(choice['outbound']))
            score = choice['delta']
            delta_N1score = choice['delta_N1']
            if VERBOSITY['brain']: 
                print(f'{n}: {score} vs {delta_N1score}')
            buckets[n] = tickets(n, score, delta_N1score, free_transfers, hesitancy_dict, quality_factor, season_avg_delta_dict[n], min_delta_dict)

        if VERBOSITY['brain']: 
            print('buckets are ', buckets)
        total = sum(buckets.values())
        if total==0: #very rare case where all transfers bad and 2 ft
            return 0
        magic_number = randint(1,total)
        #print('magic number is ', magic_number)
        bidder = -1
        while magic_number > 0:
            bidder += 1
            if bidder in buckets.keys(): 
                magic_number -= buckets[bidder]
        #print('winning bidder is ', bidder)
        return bidder


    winner = auction(choices, free_transfers, hesitancy_dict, quality_factor, season_avg_delta_dict, min_delta_dict)
    if winner == 0:
        return save_ft, choice_report
    else:
        for op in choices: #get the series that it corresponds to
            if len(safer_eval(op['outbound'])) == winner:
                return op, choice_report
    
# @return: df with close to the cheapest combination of players that is feasible
# instead just use cheapest player from 15 teams
# have to be careful in case teams have no players or injured players (like in preseason.py when we access this)
    # which kind of overcomplicated things a bit
def get_cheapest_team(full_transfer_market):
    df = full_transfer_market
    players = []
    elementssofar = [] # so don't pick same twice
    teams, index = full_transfer_market['team'].unique(), 0
    while len(players) < 15:
        position = (1 if len(players) < 2 else 2 if len(players) < 7 else 3 if len(players) < 12 else 4)
        team = teams[index]
        index = (index + 1) % len(teams) # so will loop in circles
        player = df.loc[(df['team']==team)&(df['position']==position)&(~df['element'].isin(elementssofar))].sort_values('value', ascending=True)\
            .reset_index(drop=True).iloc[0:1,:]
        if player.shape[0] == 0:
            continue
        else:
            elementssofar.append(player['element'])
            players.append(player)
    return pd.concat(players, axis=0).reset_index(drop=True)

# we work in the space of only one of the prediction lengths because need just an expected points to satisfy functions
#   and then add other one back in at the end 
# converges on a great team by making transfers
def best_wildcard_team(full_transfer_market, sell_value, bench_factor, free_hit=False,\
allowed_healths=['a'], visualize_names=False, name_df = None):
    full_team_players = get_cheapest_team(full_transfer_market)
    if free_hit:
        transfer_market = full_transfer_market.drop('expected_pts_full', axis=1)
        team_players = full_team_players.drop('expected_pts_full', axis=1)
    else:
        transfer_market = full_transfer_market.drop('expected_pts_N1', axis=1)
        team_players = full_team_players.drop('expected_pts_N1', axis=1)
    #print(team_players)
    iteration = 1
    n = 1
    # keep making 1 transfer till none good, then 2...etc till can't improve w 5 transfers
    # use shuffle to hopefully increase chance of avoidance from getting stuck loc max.
    while n < WILDCARD_DEPTH_PARAM: 
        team_players = randomly_shuffle_n_players(team_players, transfer_market, sell_value, bench_factor, 5, 13)
        scoreboard = top_transfer_options(transfer_market, team_players, sell_value, n, 1, bench_factor, 0, allowed_healths=allowed_healths)
        
        choice = scoreboard.loc[scoreboard['delta']>0] #drop extra rows in the case of not enough satisfiable transfers
        if choice.shape[0] == 0: #no improvements at this n
            n += 1
        else: #some improvement, back to 1 at a time
            #print('\nIn wildcard, n = ', n, 'iteration= ', iteration)
            n = 1

            outbound = safer_eval(choice['outbound'][0]) #the 0 on here is to choose the top choice
            inb_players = safer_eval(choice['inbound'][0])
            #print('out and in', outbound, inb_players)
            inbound = transfer_market.loc[transfer_market['element'].isin(inb_players)]
            old_players = team_players.loc[~team_players['element'].isin(outbound)]
            team_players = pd.concat([old_players, inbound],axis=0)
            #rint(team_players.sort_values('expected_pts_full', ascending=False).reset_index(drop=True)['element'].tolist())

            if visualize_names:
                transfer_option_names(choice, name_df)

        iteration += 1
        #if n > 2:
        #    print("Wildcard Iteration-", iteration, " with __", n, "__ Transfers:\n", team_players['element'].tolist())


    team_players = rename_expected_pts(team_players)
    wildcard_pts = get_points(team_players, bench_factor)

    selected_elements = team_players['element']
    final_players = full_transfer_market.loc[full_transfer_market['element'].isin(selected_elements)]
    return final_players, wildcard_pts

#return squad_selection = (elements that should be on the field, captain, vice captain)
#  captain score, bench score, 
def pick_team(team_players, health_df, with_keeper_bench=False):
    temp_players = team_players.drop('expected_pts_full', axis=1)
    temp_players = rename_expected_pts(temp_players)
    starters, bench_players = get_starters_and_bench(temp_players) #list of ints (id)
    bench_df = temp_players.loc[temp_players['element'].isin(bench_players)]
    if with_keeper_bench:
        bench_order = get_bench_order_with_keeper(bench_df)
    else:
        bench_order = get_bench_order(bench_df)

    starter_df = team_players.loc[team_players['element'].isin(starters)]
    captain, vice_captain, captain_pts = find_healthy_best_captains(starter_df, health_df)
    bench_score = team_players.loc[team_players['element'].isin(bench_players)]['expected_pts_N1'].sum()

    squad_selection = (starters, bench_order, captain, vice_captain)
    return squad_selection, captain_pts, bench_score

#@return free hit team, and points predicted for such a team
def free_hit_team(full_transfer_market, sell_value, bench_factor, allowed_healths=['a']):
    team_players, fh_points = best_wildcard_team(full_transfer_market, sell_value, bench_factor, free_hit=True, allowed_healths=allowed_healths)
    return team_players, fh_points

#@params:
# gw = week
# chip_status: dict keys=chipname, vals=0 if not played and gw played otherwise, wildcard is tuple(boolean, end_week)
# chip_max_dict: keys=chipname, vals=False if no weeks gone, otherwise max would_be_score by chip paramater
# wildcard_pts, freehit_pts, captain_pts, bench_pts: floats to be compared to max_dict
# earliest_chip_weeks: dict keys=chipname, vals = int, earliest it may be played(wildcard is tuple)
# chip_threshold_quality: float 0-2ish how much you need to beat top score by to play chip
# chip threshold_tailoff: float 0-1, how quick we decrease chip_threshold_quality
#@return: 'wildcard', 'freehit', 'bench_boost', 'triple_captain', or 'normal'
def play_chips_or_no(gw, chip_status, chip_threshold_dict, wildcard_pts, freehit_pts, captain_pts,\
          bench_pts, earliest_chip_weeks, chip_threshold_tailoffs):
    def tailoff_coeff(gw, tailoff, last_gw):
        eps = .00001#avoid divide by 0
        return 1-(factorial(gw)/factorial(last_gw))**tailoff + eps

    this_week_chip_scores = {
        'wildcard': wildcard_pts,
        'freehit': freehit_pts,
        'bench_boost': bench_pts,
        'triple_captain': captain_pts
    }
    if VERBOSITY['brain_important']: 
        print('this week chip scores: ', this_week_chip_scores, '\n chip thresholds: ', chip_threshold_dict)
        print(earliest_chip_weeks)
    chip_qualities = {}
    for i, chip in enumerate(['wildcard', 'freehit', 'bench_boost', 'triple_captain']):
        last_gw = 38
        if chip =='wildcard': 
            last_gw = chip_status[chip][1]
        status = (chip_status[chip][0] if chip=='wildcard' else chip_status[chip])
        threshold = chip_threshold_dict[chip]
        if VERBOSITY['brain']: 
            print(chip_status, status, threshold, chip_threshold_dict)
        if threshold == 0:
            threshold += .00001 # (to avoid divide by 0)
        if not threshold: #Got a False because we didn't have any data for this chip yet
            continue
        score_to_beat = threshold*tailoff_coeff(gw, chip_threshold_tailoffs[i], last_gw)
        print(tailoff_coeff(gw, chip_threshold_tailoffs[i], last_gw))
        print(chip_threshold_tailoffs[i], score_to_beat)
        week_score = this_week_chip_scores[chip]
        if week_score > score_to_beat:
            print('beat a')
            if not(status):
                print('beat b')
                if earliest_chip_weeks[chip] <= gw:
                    print('beat c')
                    chip_qualities[chip] = week_score/score_to_beat 
    
    if chip_qualities == {}:
        return 'normal'
    else:
        return max(chip_qualities, key= lambda x: chip_qualities[x])

# returns sub in sub out tuple for input to agent
def figure_out_substitution(on_field, on_bench, starters, bench):
    send_to_bench = set(bench).difference(set(on_bench)) 
    send_to_field = set(starters).difference(set(on_field))
    if len(send_to_bench) != len(send_to_field):
        raise Exception("our logic is somehow wrong, or got wrong values")

    return list(send_to_field), list(send_to_bench)

# this will order the two lists so there is a 1 to 1 correspondance between the positions 
def match_positions(inbound_list, outbound_list, full_transfer_market):
    if VERBOSITY['brain']: 
        print('at the beginning: ', inbound_list, outbound_list)
    proper_inbound = []
    for player in outbound_list:
        position1 = full_transfer_market.loc[full_transfer_market['element']==player]['position'].tolist()[0]
        for other_player in inbound_list:
            position2 = full_transfer_market.loc[full_transfer_market['element']==other_player]['position'].tolist()[0]
            if position1 == position2:
                proper_inbound.append(other_player) 
                inbound_list.remove(other_player)
                break
    
    if VERBOSITY['brain']: 
        print('at the end: ', proper_inbound, outbound_list)
    return proper_inbound, outbound_list

