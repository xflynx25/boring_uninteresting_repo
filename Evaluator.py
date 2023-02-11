"""
STATE MACHINE
f (season, starting_team, g(processed_data, current_team, sell_value, free_transfers, chip_availabilities)) 
    --> season score 
!-! We keep track of team as what we bought them for, recalculate their sell price before send to transfer func
"""
#!%%
print('first thing on the page')
from private_versions.constants import DROPBOX_PATH, WILDCARD_2_GW_STARTS, change_wildcard_depth, POINT_HISTORY_REFERENCE_PATH

change_wildcard_depth(4) # 4 for now because just speed running for evaluator, 5 for actual

from general_helpers import safe_make_folder, blockPrint, enablePrint
from random import random
from Evaluator_helpers import this_week_chip, get_meta_gwks_dfs, get_data_df, CENTURY, get_league_player_starting_team,\
    get_athena_skeleton, compute_all_transfer_dfs, generate_x_random_starting_teams, MALLEABLE_PARAMS, evolve_team,\
        id_to_teamname_converter_saved, prepare_for_pretty_print, MIXED_MODELS, stringify_if_listlike
import pandas as pd
from statistics import stdev
import numpy as np 
from os import listdir
import time
import random
import tempfile
import Overseer_Simulator
FPL_AI = Overseer_Simulator.FPL_AI
from config.folder_setup import initialize_overseer_folders
from printing import pretty_print_gw
from general_helpers import safe_read_csv, safer_eval



# @param: 
##  - starting_team is list of elements 
##  - transfer func ^^^^
##  - when transfer is early, late, or random (where can be any between the two weeks, may help us because this is uncertain knowledge for comp)
##  - print_transfers_over is for debugging in other applications, we print out wks with at least x transfers
# squad is list of (element, sell_value)
# team is df of element, buy price, current_price, sell_price
# we keep track of buy prices, and then calculate other 
def simulate_season(data_df, starting_team, gw1_pick_team, transfer_function, when_transfer='late', starting_sv = 100.0 * 10, print_transfers_over=16,\
        verbose=False):
    if type(when_transfer) == list:
        EARLY_WEEKS, LATE_WEEKS = when_transfer
    def current_price(price_data, x):
        price_np = price_data.loc[price_data['element']==float(x['element'])]['value'].to_numpy()
        return (x['purchase'] if len(price_np) == 0 else price_np[0])
    seasons = data_df['season'].unique()
    if len(seasons) > 1:
        raise Exception('data df more than 1 season')
    else:
        SEASON = seasons[0]
    name_df = data_df[['element', 'name']].drop_duplicates() #using for printing

    ## SETUP ##
    # get the earliest prices for each player, and also have the element columns, need to do this in case people don't play gw1
    starting_info = []
    for player in data_df['element'].unique():
        start_pos, start_price = data_df.loc[data_df['element']==player][['position','value']].to_numpy()[0]
        starting_info.append([player, start_pos, start_price])
    starting_info_df = pd.DataFrame(starting_info, columns=['element', 'position', 'value'])
    team = pd.DataFrame(index=range(len(starting_team)),columns=['element','position', 'purchase','now','sell'])
    team.loc[:, 'element'] = starting_team
    team.loc[:, 'position'] = team['element'].apply(lambda x: starting_info_df.loc[starting_info_df['element']==float(x)]['position'].to_numpy()[0])
    team.loc[:, 'purchase'] = team['element'].apply(lambda x: starting_info_df.loc[starting_info_df['element']==float(x)]['value'].to_numpy()[0])
    sv = starting_sv
    itb = sv - team['purchase'].sum()
    ft = 1
    chip_status = {'wildcard': [0, WILDCARD_2_GW_STARTS[SEASON]], 'freehit': 0, 'bench_boost': 0, 'triple_captain': 0}
    scores = {}
    temp_storage = None # Freehit team
    print('about to block print')
    blockPrint()
    for gw in range(1,39):
        print(f'gw is {gw}')
        SKIP_TO_WEEK = 2
        if gw < SKIP_TO_WEEK and gw != 1: # to skip to certain week
            scores[gw] = random.randint(20,80)
            continue
        ## RESTORE FREE HIT ## 
        if temp_storage is not None: 
            team, itb = temp_storage
            temp_storage = None
        ## Adjust Prices ##
        when_transfer_this_week = (when_transfer if type(when_transfer) == str else 'early' if gw in EARLY_WEEKS else 'late') #default to late
        this_gw_data = data_df.loc[data_df['gw']==gw]
        if when_transfer_this_week == 'early' and gw != 1:
            price_data = data_df.loc[data_df['gw']==gw-1]
        elif when_transfer_this_week == 'late' or gw == 1:
            price_data = this_gw_data
        else:
            print('when_transfer, gw', when_transfer_this_week, gw)
            price_data = 0
            raise Exception("soemhow price data wasn't defined")
        team.loc[:, 'now'] = team.apply(lambda x: current_price(price_data, x), axis=1)
        team.loc[:, 'sell'] = (team['now'] - team['purchase'])
        team.loc[:, 'sell'] = (2 * team['purchase'] + team['sell'].apply(lambda x: [x if x >= 0 else 2*x][0]) ) // 2
        sv = team['sell'].sum() + itb
    

        ## TRANSFER ##
        if gw > 1:

            ## DECISION MAKING ## 
            blockPrint()
            squad = team[['element', 'sell']].to_numpy().tolist()
            transfer_args = (this_gw_data, gw, squad, sv, ft, chip_status, scores)
            transfer, chip, captain, vcaptain, bench = transfer_function(transfer_args)
            if verbose: 
                print(transfer, chip, captain, vcaptain, bench)
            inbound, outbound = transfer 
            
            ## FREE HIT BUSINESS ## 
            if chip == 'freehit':
                temp_storage = team.copy(), itb


            ## EVOLVE TEAM ##
            team = evolve_team(team, inbound, outbound, price_data)
            enablePrint()
            if chip == 'wildcard':
                chip_status['wildcard'][0] = gw
            elif chip:
                chip_status[chip] = gw
            if gw == WILDCARD_2_GW_STARTS[SEASON] - 1:
                chip_status['wildcard'] = [0, 38]
            itb = sv - team['sell'].sum()
            ft, hit = [(1, 0) if chip in ('wildcard', 'freehit') else (min(max(ft - len(inbound), 0)+1, 2) , (len(inbound)>ft)*4*(len(inbound)-ft))][0]
            #if len(inbound) > print_transfers_over and verbose:
            #    print(f'gw{gw}: ntransfers = {len(inbound)}')
            print('gw: ', gw, 'team_shape = ', team.shape, '  ft: ', ft, '   round transfers was ', len(inbound), '   any chips = ', chip, '   hit= ', hit)
        else:
            chip, captain, vcaptain, bench = gw1_pick_team
            inbound, outbound = set(), set()
            hit = 0

        if hit != 0:
            print(f'gw{gw} took {hit} point hit')

        
        ## EVALUATE THE TEAM ##
        results = this_gw_data[['element', 'position', 'minutes_N1', 'total_points_N1']]
        score = score_round(results, team, chip, captain, vcaptain, bench, hit)
        scores[gw] = score
        if verbose:
            print(f'\n\ngw{gw}: {score}  ------- chip={chip}  |  sv={sv}  |  itb={itb}  |  ft={ft}\n')
            if verbose == 'full':
                wk_transfer_names_and_points, field_name_df, bench_name_df, captain_name, vcaptain_name=\
                    prepare_for_pretty_print(season, gw, data_df, name_df, team, bench, (inbound, outbound), captain, vcaptain)
                pretty_print_gw(gw, wk_transfer_names_and_points, field_name_df, bench_name_df, captain_name, vcaptain_name,\
                    chip, score, sum(scores.values()), hit,benchmark_path=POINT_HISTORY_REFERENCE_PATH)


    return scores

    
#@ return some_dict 
#@ param: bench is a list of elements
# ####### team is df which has element and position
# ####### results is the gw df with just [['element', 'position', 'minutes_N1', 'total_points_N1']]
def score_round(results, team, chip, captain, vcaptain, bench_list, hit):
    bench_set = {x for x in bench_list}
    score = 0

    nonminuteless = set(results.loc[results['minutes_N1']>0]['element'].to_list()) #have to do this way bcz of players who join late
    team_minuteless = set(team['element'].to_list()).difference(nonminuteless)
    field_minuteless = team_minuteless.difference(bench_set)
    bench_minuteless = team_minuteless.intersection(bench_set)

    ## CAPTAIN
    if captain in field_minuteless:
        captain = vcaptain
    score += results.loc[results['element']==captain]['total_points_N1'].to_list()[0]
    if chip =='triple_captain':
        score *= 2

    score += sum(results.loc[results['element'].isin(team['element'])]['total_points_N1'])
    
    if chip != 'bench_boost':
        fields = team.loc[~team['element'].isin(bench_set)]['position'].to_list()
        outs = team.loc[team['element'].isin(field_minuteless)]['position'].to_list()
        bench_set = bench_set.difference(bench_minuteless)
        bench_list = [x for x in bench_list if x in bench_set]
        bench_pos = [team.loc[team['element']==x]['position'].to_list()[0] for x in bench_list]
        
        fields_dict, out_dict = {},{}
        for i in fields:
            fields_dict[i] = fields_dict.get(i, 0) + 1
            
        def requirement(pos):
            if pos == 1:
                return 1
            elif pos == 2: 
                return 3
            elif pos == 3:
                return 0
            elif pos == 4:
                return 1
            else:
                raise Exception("Invalid Position")

        # get valid subs for the position
        for sub, sub_pos in zip(bench_list.copy(), bench_pos):
            found = False
            for pos in outs:
                if sub_pos == 1 and pos != 1: #keeper for keeper
                    continue
                if pos == sub_pos or fields_dict[pos] > requirement(pos):
                    bench_list.remove(sub)
                    found = pos
                    break
            if found:
                outs.remove(found)
                fields_dict[pos] -= 1
                fields_dict[sub_pos] = fields_dict.get(sub_pos, 0) + 1

        score -= sum(results.loc[results['element'].isin(bench_list)]['total_points_N1'])
    return score - hit



#@return: transfer, chip, captain, vcaptain, bench
def some_transfer_function(data, gw, squad, sv, ft, chip_status, weekly_scores):
    pass


# season is 2021 format
#@param: personality is the dict, when_transfer = ('early', 'late', 'mixed'?) if mixed, transfer switchup
# record the params, and first 10, next 10, next 10, last 8, and total
# we should actually just compute the transfer markets once for each set of models, ~ 7Kb each 
def get_athena_season(season, starter_packs, personalities, verbose=False):
    
    data_df = get_data_df(CENTURY, season)
    all_field_suites = set([item for suite_list in [x['field_model_suites'] for x in personalities] for stage_suites in suite_list for item in stage_suites ])
    all_keeper_suites = set([item for suite_list in [x['keeper_model_suites'] for x in personalities] for stage_suites in suite_list for item in stage_suites])
    print( all_field_suites, all_keeper_suites)
    tm_folder = DROPBOX_PATH + f"Simulation/athena_Simulation/transfer_markets/{season}/"
    safe_make_folder(tm_folder) # make sure we hav e afolder for simulating this year
    all_field_suites = [x for x in all_field_suites if x+'.csv' not in listdir(tm_folder)]
    all_keeper_suites = [x for x in all_keeper_suites if x+'.csv' not in listdir(tm_folder)]
    all_suites = (all_field_suites, all_keeper_suites)
    print(all_suites,listdir(tm_folder))
    #raise Exception('suite chekc')
    compute_all_transfer_dfs(season, tm_folder, data_df, all_suites)
    

    stats = {}
    for i, personality_og in enumerate(personalities): 
        print('Personality ', i+1, '  ')
        personality = personality_og.copy()
        personality['field_model_suites'] = [[pd.read_csv(tm_folder + model + '.csv') for model in stage] for stage in personality['field_model_suites']]
        personality['keeper_model_suites'] = [[pd.read_csv(tm_folder + model + '.csv') for model in stage] for stage in personality['keeper_model_suites']]

        these_scores = {'1_9':[],'10_19':[],'20_29':[],'30_38':[],'7_33':[],'total':[]}
        for start_team, start_pick_team in starter_packs:
            print('a Trial')
            #print(start_team, start_pick_team)
            with tempfile.TemporaryDirectory() as directory:
                personality['folder'] = directory
                ai = FPL_AI(**personality) #using the same ai for all in these starter packs but need to change the folder so we put it in here
                athena_transfer_function = lambda x: ai.make_moves(*x)

                initialize_overseer_folders(personality['folder'])

                start = time.time()
                score = simulate_season(data_df, start_team, start_pick_team, athena_transfer_function, when_transfer=personality['when_transfer'], starting_sv=1000.0, verbose=verbose)
                print("Runtime = ", time.time() - start)
                s = list(score.values())
                these_scores['1_9'].append(sum(s[:10]))
                these_scores['10_19'].append(sum(s[10:20]))
                these_scores['20_29'].append(sum(s[20:30]))
                these_scores['30_38'].append(sum(s[30:]))
                these_scores['7_33'].append(sum(s[7:34])) #when we can use the full 6 either side
                these_scores['total'].append(sum(s))

            
        # the statistics we want to keep (1-10, 11-20, 21-30, 31-38, mean, min, 1/4, 3/4, max)
        total = these_scores['total']
        print('Total Points: ', total)
        stats[i] = dict({key:sum(val)/len(val) for (key, val) in these_scores.items()}, **{'stderr':round(stdev(total)/(len(starter_packs)**(1/2)), 2), 'min':min(total), 'quantile_1':np.quantile(total, .25),'quantile_3':np.quantile(total, .75),'max':max(total)})
        print('These stats: ', stats[i])
    return stats 


"""
    'max_hit'
    'bench_factors'
    'value_vector'
    'num_options'
    'quality_factor'
    'hesitancy_dict'
    'min_delta_dict'
    'earliest_chip_weeks'
    'chip_threshold_construction'
    'chip_threshold_tailoff'
    'player_protection'
    'field_model_suites'
    'keeper_model_suites' 
""" 
###### RUNTIME:: 37 minutes to simulate 1 season
# @param: season: (2021), num_trials: how many diff starting teams to try (random from top 10,000)
    # testing: dict of those to vary, values are list of ones to check
# __note__ ;; if 'when_transfer' is early, we automatically change the features to field_early_transfer_batch
# @return: None, save with cols: num_trials, 13 features, 9 results 
def athena_param_sweep(season, num_trials, testing, verbose=False):
    print(f'season is {season}')
    # setup the datastructures
    SEED  = random.randint(1, 10**9)
    ATHENA_SKELETON = get_athena_skeleton(season)
    starter_packs, _ = generate_x_random_starting_teams(season, num_trials)
    
    # if the testing has all things even those that aren't varying
    def get_all_testing_combos(groups_left, dicts_built = [{}]):
        if len(groups_left) == 0:
            return dicts_built

        key = list(groups_left.keys())[0]
        val_list = groups_left.pop(key)

        new_dicts = []
        for prev_dict in dicts_built:
            for val in val_list:
                temp_dict = prev_dict.copy()
                temp_dict.update({key:val})
                new_dicts.append(temp_dict)
        
        return get_all_testing_combos(groups_left, dicts_built=new_dicts)

    def get_all_combos(testing):
        personalities = []
        testing_combos = get_all_testing_combos(testing)
        for combo in testing_combos:
            pers = ATHENA_SKELETON.copy()
            for key, val in combo.items():
                pers[key] = val
            personalities.append(pers)
        return personalities
            
    def adjust_based_on_when_transfer(personalities):
        finals = []
        for personality in personalities:
            # if we want to do a sweep on models but make most do late transfers 
            if 'field_early_transfer_batch' in personality['field_model_suites'][0] and personality['when_transfer'] == 'late':  
                personality['when_transfer'] = 'early'
            wt = personality['when_transfer']
            if type(wt) == list:
                personality['field_model_suites'], personality['keeper_model_suites'] = MIXED_MODELS
            if wt == 'early':
                personality['field_model_suites'] = [['field_early_transfer_batch']]
            elif wt == 'mixed':
                pass
            elif wt == 'late':
                pass
            finals.append(personality)
        return finals
    
    personalities = get_all_combos(testing.copy()) # need to combine with athena skeleton already so mahybe just allow for only puttting the things we are going to chagne.
    personalities = adjust_based_on_when_transfer(personalities)
    print(f'season is {season}')
    print(personalities)


    # get data & turn to dataframe
    print('about to get the athena seasons')
    stats = get_athena_season(season, starter_packs, personalities, verbose=verbose) #output info 
    param_choices = [{key: stringify_if_listlike(val) for (key, val) in personality.items() if key in MALLEABLE_PARAMS} for personality in personalities] #input info
    combined_info = {i: dict(stats[i], **param_choices[i]) for i in range(len(personalities))}
    df = pd.DataFrame.from_dict(combined_info, orient='index')
    df['num_trials'] = num_trials
    df['seed'] = SEED
    df['season'] = season
    print(f'season is {season}')

    # save all
    path = DROPBOX_PATH + "Simulation/athena_Simulation/param_sweep/data.csv"
    old = safe_read_csv(path)
    pd.concat([old, df]).reset_index(drop=True).to_csv(path)
    return df#for now


#@return: transfer, chip, captain, vcaptain, bench
def top_player_transfer_function(data, gw, squad, sv, ft, chip_status, weekly_scores, meta_df, gwks_df):
    gwk_df = gwks_df.loc[gwks_df['gw']==gw]
    chip = this_week_chip(gw, meta_df)
    if 'wildcard' in chip:
        chip = 'wildcard'
    if chip == 'free_hit':
        chip = 'freehit'
    transfer = [eval(x) for x in gwk_df[['inb', 'outb']].to_numpy()[0]]
    captain, vcaptain = gwk_df[['captain', 'vcaptain']].to_numpy()[0]
    bench = gwk_df[[f'bench{i}' for i in range(1,5)]].to_numpy()[0].tolist()
    return transfer, chip, captain, vcaptain, bench



# @param: season is 2021 format
def get_league_player_weekly_scores(season, ranks, league = 'Overall', interval = 250, print_transfers_over=16):
    ''' getting the season dataset for all the footballers ''' 
    data_df = get_data_df(CENTURY, season)

    competitor_scores = {}
    for rank in ranks: 
        #print('getting rank ', rank)
        ''' getting the starting team and transfer func '''
        meta_df, gwks_df = get_meta_gwks_dfs(season, league, interval, rank)
        competitor_tf = lambda x: top_player_transfer_function(*x, meta_df, gwks_df)
        competitor_starting_team, competitor_pick_team = get_league_player_starting_team(season, rank, league, interval)
        
        ''' running '''
        #start = time.time()
        competitor_score = simulate_season(data_df, competitor_starting_team, competitor_pick_team, competitor_tf, starting_sv=1500.0, print_transfers_over=print_transfers_over)
        competitor_scores[rank] = competitor_score
        #print("Runtime = ", time.time() - start)
    return competitor_scores





if __name__ == '__main__':
    ''' VERIFYING SCORES OF PREVIOUS SEASONS ''' """
    #scores = get_league_player_weekly_scores(2021, list(range(1,8)), league='White Hart Kane', interval=7)
    nbad, nexcept = 0,0
    #for rank, scorey in scores.items():
    for rank in range(1,1001):
        try:
            scorey = get_league_player_weekly_scores(2021, [rank])[rank]
            meta_df, gwks_df = get_meta_gwks_dfs(2021, 'Overall', 250, rank)
            actual = meta_df.loc[meta_df['rank']==rank]['total_points'].to_list()[0]
            computed = sum(scorey.values())
            if actual != computed:
                print(rank, ' actual: ', actual, '  --   computed: ', int(computed))
                #print({key: int(val) for key,val in scorey.items()})
                nbad += 1
        except:

            print(rank, ' exception')
            nexcept += 1
    print(nbad, '  ', nexcept)
    """
    #point_spread = get_league_player_weekly_scores(2021, [1], league = 'White Hart Kane', interval = 7, print_transfers_over=16)[1]
    #print(f"matthew total points:{sum(point_spread.values())} \n{point_spread}")
    #df = pd.DataFrame([[a,b] for a,b in point_spread.items()], columns=['gw','points'])
    #df.to_csv(DROPBOX_PATH + '/Human_Seasons/Reference_Point_Markers/mat_the_w.csv')
    #raise Exception()
    ''' PERSONALITY SEARCHING & OPTIMIZATION ''' """
    #athena_param_sweep(season, num_trials, testing, when_transfer=['late'])
    # or first just od on the past seasons personalities
    from Personalities import Athena_v10a
    season, num_trials = 1718, 4
    starter_packs, ranks = generate_x_random_starting_teams(season, num_trials)
    personality = get_athena_skeleton(season)#Athena_v10a
    personality['field_model_suites'] = [['field_early_transfer_batch'],['field_all_batch']]#  [['field_all_batch']]
    personality['keeper_model_suites']= [['keeper_all_batch'],['keeper_all_batch']]
    personality['folder'] = DROPBOX_PATH + 'Simulation/Athena-v1.0a/'
    personality['when_transfer'] = [list(range(1,19)), list(range(19, 39))] #'early'
    
    this_rank = ranks[-1]
    #point_spread = get_league_player_weekly_scores(2021, [this_rank])[this_rank]
    #print(f"our opponent total points:{sum(point_spread.values())}\n{point_spread}")
     

    personalities =  [personality]
    # get data & turn to dataframe
    stats = get_athena_season(season, starter_packs, personalities, verbose='full') #output info 
    param_choices = [{key: val for (key, val) in personality.items() if key in MALLEABLE_PARAMS} for personality in personalities] #input info
    combined_info = {i: dict(stats[i], **param_choices[i]) for i in range(len(personalities))}
    df = pd.DataFrame.from_dict(combined_info, orient='index')
    df['num_trials'] = num_trials
    print(df)

    #reference_player = get_league_player_weekly_scores(season, [1], league = 'White Hart Kane', interval = 7, print_transfers_over=16)
    #print(f'matthew total points: {sum(reference_player[1].values())}')

    # comparing athena to matthews scores 
    
    """
    '''
    
    personalities = [get_athena_skeleton(season)]

    
    this_rank = ranks[-1]
    point_spread = get_league_player_weekly_scores(2021, [this_rank])[this_rank]
    print(f"our opponent total points:{sum(point_spread.values())}\n{point_spread}")
     

    personalities =  [personality]
    # get data & turn to dataframe
    stats = get_athena_season(season, starter_packs, personalities, 'late', verbose=True) #output info 
    param_choices = [{key: val for (key, val) in personality.items() if key in MALLEABLE_PARAMS} for personality in personalities] #input info
    combined_info = {i: dict(stats[i], **param_choices[i]) for i in range(len(personalities))}
    df = pd.DataFrame.from_dict(combined_info, orient='index')
    df['num_trials'] = num_trials
    print(df)
    '''

    ''' The default sweep when you are testing the first model trained well - Should score crazy points like 2500+'''
    season, num_trials = 2122, 2 # 5 mins * 2 * 4 = 40 mins
    testingdefault = {'field_model_suites': [[['full_squared_error']], [['full']]], 
                'bench_factors' : [(.075,.0075), (0.1, 0.01)]
            }
    print('ABOUT TO CALL')
    df = athena_param_sweep(season, num_trials, testingdefault)

    raise Exception('done')
    
    ''' actually doing param sweeps ''' 
    mixed = [list(range(1,19)), list(range(19, 39))]
    season, num_trials = 2021, 20
    testing1 = {
        'when_transfer':(mixed, 'early', 'late'), 
        'earliest_chip_weeks' : ({'wildcard':(7, 15), 'freehit': 15, 'triple_captain': 15, 'bench_boost':15},\
            {'wildcard':(4, 15), 'freehit': 15, 'triple_captain': 15, 'bench_boost':15})
    }
    #df = athena_param_sweep(season, num_trials, testing1)
    season, num_trials = 2021, 15 # currently, can't really test on previous seasons because was trained on this and does amazing
    testing2 = {
        'bench_factors': [(.15,.015),(.1, .015)],
        'value_vector': ([.3,.3,.4],[.05,.7,.25],[0,0,1]),
        'quality_factor': [2, 3.5, 5],
    }
    #df = athena_param_sweep(season, num_trials, testing2)
    season, num_trials = 2021, 25 
    testing3 = {
        'bench_factors': [(.1,.015),(.05, .01)],
        'value_vector': [[.1,.6,.3]],
        'quality_factor': [3],
        'hesitancy_dict': [
            {1: {0: .4, 1: .3, 2: .5, 3:.6}, 2: {1: .4, 2: .3, 3: .5, 4:.6}}, # classic
            {1: {0: .3, 1: .3, 2: .6, 3:.7}, 2: {1: .3, 2: .3, 3: .6, 4:.7}}, # conservative
            {1: {0: .4, 1: .4, 2: .45, 3:.55}, 2: {1: .4, 2: .4, 3: .45, 4:.55}}, # aggressive
            {1: {0: .5, 1: .5, 2: .99, 3:.99}, 2: {1: .5, 2: .5, 3: .99, 4:.99}}, # no hits
        ]
    }
    #df = athena_param_sweep(season, num_trials, testing3)
    season, num_trials = 2021, 2
    testing4 = {
        # BEST SCORER SO FAR #
        'max_hit' : [0],
        'hesitancy_dict': [
            {1: {0: .5, 1: .5}, 2: {1: .5, 2: .5}}, # no hits
        ], 
        'bench_factors': [(.1,.015)],
        'value_vector': [[.1,.6,.3]],

        # TESTING #
        #'field_model_suites': [[['no_ict', 'no_ict_transfers_price', 'no_ictANY_transfers_price']],[['field_all_batch']], [['full']], \
        #    [['individuals', 'sparse_individuals']], [['early', 'late']], [['onehot', 'priceless', 'dgw']], [['no_transfer_info', 'onehot']],\
        #    [['field_early_transfer_batch']]],
        'field_model_suites': [[['full'], ['full_squared_error'], ['onehot']]]
    }
    # now to see if the new models are workingrning)
    '''This is in here because we only have a few pretrains'''
    df = athena_param_sweep(season, num_trials, testing4)

    testing_athena = {
        'max_hit' : [8], #int, positive, for now we know we can't afford 4 transfers 
        'bench_factors' : [(.075,.0075)],
        'value_vector' : [[0.15, 0.35, 0.5]],# ['a', 'd']], #listlike - [worth, next_match_delta, full_match_delta]
        'num_options' : [12], #int - how many to consider in transfer search single n
        'quality_factor' : [4.5], #float - multiplier for relative quality of rank-n top transfer
        'hesitancy_dict' : [{
            1: {0: 0.4, 1: 0.3, 2: 0.6, 3: 0.7},
            2: {1: 0.4, 2: 0.3, 3: 0.6, 4: 0.7}
        }], #dict - keys:ft values=dict with keys=num_transfers, vals = 0-1 float 
        'min_delta_dict': [{
            1: {0: 0, 1: .25, 2: 1, 3:1.75},
            2: {0: 0, 1: 0, 2: .75, 3: 1.5, 4:2.25}
        }],
        'chip_threshold_tailoffs' : [[.3,.19,.19,.19]], #float - 0-1, .2 tails off around gw34, .1 around gw30, .05 around gw20, slow tailoff
        'field_model_suites':  [
            [['dgw']], 
            [['full']], 
            [['full_squared_error']], 
            [['onehot']], 
            [['priceless']], 
            [['no_ict']], 
            [['no_ict_transfers_price']], 
            [['no_ictANY_transfers_price']], 
            [['no_transfer_info']], 
            [['individuals']], 
            [['sparse_individuals']],  
        ]    
        }

    df = athena_param_sweep(season, 24, testing_athena)


    """ INTERESTING COMBOS"""
    testing_athena['field_model_suites'] = [
            [['field_all_batch']], 
            [['full', 'full_squared_error','onehot', 'priceless', 'no_ict',\
                'no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info']], 
            [['full', 'full_squared_error','onehot','early', 'late','priceless',\
                'no_ict', 'no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info',\
                'dgw']], 
            [['full', 'full_squared_error','onehot','priceless',\
                'no_ict', 'no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info',\
                'dgw', 'individuals', 'sparse_individuals']],
            [['full', 'full_squared_error','onehot', 'early', 'late','priceless',\
                'no_ict', 'no_ict_transfers_price', 'no_ictANY_transfers_price', 'no_transfer_info',\
                'individuals', 'sparse_individuals']],
            [['full', 'full_squared_error','onehot','early', 'late', 'individuals']],
            [['full', 'early', 'late','priceless',\
                'no_ict', 'no_ictANY_transfers_price', 'individuals']],
            [['early', 'late','priceless',\
                'no_ict', 'no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info',\
                'individuals', 'sparse_individuals']],
            [['full']], [['no_ict', 'priceless', 'individuals']], [['full_squared_error', 'early', 'late', 'priceless', 'dgw']]
            ]
            
    df2 = athena_param_sweep(season, 24, testing_athena)
    


    """ Keeper Combos"""
    testing_athena['field_model_suites'] = [
            [['field_all_batch']]
            ]    

    testing_athena['keeper_model_suites'] = [
            [['keeper_engineering']],
            [['keeper_engineering_squared_error']],
            [['keeper_extra_crossval']],
            [['keeper_no_price_mse']],
            [['keeper_all_batch']], 
            [['keeper_engineering','keeper_no_price_mse']],
            [['keeper_extra_crossval','keeper_engineering_squared_error']],
    ]
            
    df3 = athena_param_sweep(season, 30, testing_athena)
# %%
