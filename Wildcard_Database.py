#%%
from constants import DROPBOX_PATH, TM_FOLDER
import pandas as pd
    

from general_helpers import get_columns_containing, get_data_df, get_meta_gwks_dfs
from Evaluator import get_league_player_weekly_scores
from Evaluator_helpers import evolve_team_no_price_info
#from Evaluator_helpers import KEEPER_MODELS, FIELD_MODELS
from Overseer_Simulator import overseer_simulator_get_transfer_market
from Oracle_helpers import train_rf_with_threshold, save_model, load_model
from Oracle import create_wildcard_datapoint
import math
import time
import os
SAVED_WILDCARD_TRAINING_PATH = DROPBOX_PATH + 'Human_Seasons/saved_wildcard_data.csv'
#DATA_DF = get_data_df(20, 2021)
#META_DF1, GWKS_DF1 = get_meta_gwks_dfs(2021, 'Overall', 250, 1)
#LEAGUE_PLAYER_SCORES1 = get_league_player_weekly_scores(2021, [1], league = 'Overall', interval = 250)[1]


""" ### %%% $$$ ^^^ HELPERS ^^^ $$$ %%% ### """
def top_player_create_transfer_market(data_df, name_df, field_models, keeper_models, gw):
    player_injury_penalties = {x:1 for x in data_df.loc[data_df['gw']==gw]['element'].unique()}
    field_model_df = [pd.read_csv(TM_FOLDER + model + '.csv', index_col=0) for model in field_models]
    keeper_model_df = [pd.read_csv(TM_FOLDER + model + '.csv', index_col=0) for model in keeper_models]
    gw_tm = overseer_simulator_get_transfer_market(gw, field_model_df, keeper_model_df, [],\
        [], [[], player_injury_penalties, set(data_df.loc[(data_df['gw']==gw)&(data_df['FIX1_num_opponents']==0.0)]['element'].to_list())], name_df)
    return gw_tm
      
""" ### %%% $$$ ^^^ END HELPERS ^^^ $$$ %%% ### """



# RUNTIME: data extraction :: 40s / person (90 per hour) 
#    && for training these it takes 
# @param: wildcard_end_seasons should mirror the seasons in the week it ends
def train_wildcard_predictor(seasons, wildcard_end_seasons, n_per_season, when_transfer = 'late'):
    BENCH_FACTOR = .15
    if when_transfer == 'late':
        field_models, keeper_models = ['field_all_batch'], ['keeper_all_batch']
    elif when_transfer == 'early':
        field_models, keeper_models = ['field_early_transfer_batch'], ['keeper_all_batch']
    try:
        SAVED_DF = pd.read_csv(SAVED_WILDCARD_TRAINING_PATH, index_col=0)
        ALREADY_DONE = {(season, int(rank)) for season, rank in SAVED_DF[['season', 'rank']].to_numpy()}
        MAX_ALREADY_DONE = (SAVED_DF['rank'].max() if len(ALREADY_DONE) > 0 else 0)
    except:
        SAVED_DF, ALREADY_DONE,MAX_ALREADY_DONE = pd.DataFrame(), set(), 0

    X, y = [],[]
    # PREPARE TRAINING SET
    for season, wildcard_end in zip(seasons, wildcard_end_seasons):
        data_df = get_data_df(20, season)#DATA_DF#
        name_df = data_df[['element', 'name']].drop_duplicates()
        fix_df = pd.read_csv(DROPBOX_PATH + f"20{str(season)[:2]}-{str(season)[2:]}/fix_df.csv", index_col=0)
        failed_in_a_row, rank = 0, 1
        while(failed_in_a_row < 10 and rank <= n_per_season):
            try:
                print(f'trying rank {rank}')
                if (season, rank) in ALREADY_DONE:
                    df = SAVED_DF.loc[(SAVED_DF['rank']==rank)&(SAVED_DF['season']==season)]
                    failed_in_a_row = 0
                else:
                    ''' RIGHT NOW ASSUMING IF WE HAVE DONE SOMEONE HIGH THEN WE HAVE DONE THE LOWER SO NO NEED TO CHECK THEM AGAIN FOR CORRECTNESS '''
                    if MAX_ALREADY_DONE > rank:
                        rank += 1
                        continue

                    start = time.time()
                    meta_df, gwks_df = get_meta_gwks_dfs(season, 'Overall', 250, rank)
                    wc1, wc2, fh, bb, tc = [int(x) for x in meta_df[['wildcard1', 'wildcard2', 'free_hit', 'bench_boost', 'triple_captain']].to_numpy()[0]]
                    if math.isnan(wc1) or math.isnan(wc2) or int(wc2) == 2: #didn't play both wc within our constraints (not gw2)
                        rank += 1
                        continue
                    
                    points_each_week = get_league_player_weekly_scores(season, [rank], league = 'Overall', interval = 250)[rank]
                    if sum(points_each_week.values()) != meta_df.loc[meta_df['rank']==rank]['total_points'].to_list()[0]:
                        raise Exception("Computed points not equalling the recorded points")

                    team = set(meta_df[[f'player_{i}' for i in range(1,16)]].to_numpy()[0])
                    ft = 0
                    datapoints = {}
                    for gw in range(1,39): #only weeks where option to play
                        #print(f'gw{gw}')
                            
                        ## CREATE DATAPOINT
                        if  gw in list(range(3, wc1+1)) + list(range(wildcard_end, wc2+1)): #only weeks where option to play
                            gw_tm = top_player_create_transfer_market(data_df, name_df, field_models, keeper_models, gw)
                            chip_played_dict = {'freehit':fh, 'triple_captain':tc, 'bench_boost':bb}
                            bench = set(gwks_df.loc[gwks_df['gw']==gw][[f'bench{i}' for i in range(1,5)]].to_numpy()[0])
                            datapoint = create_wildcard_datapoint(data_df.loc[data_df['gw']==gw], fix_df, bench, points_each_week, gw_tm, wildcard_end, BENCH_FACTOR, chip_played_dict, team, ft, gw)
                            datapoint['did_wc'] = gw in (wc1, wc2)
                            datapoints[gw] = datapoint

                        ## EVOLVE TEAM
                        if gw != 1 and gw != fh:
                            inb, outb = [eval(x) for x in gwks_df.loc[gwks_df['gw']==gw][['inb', 'outb']].to_numpy()[0]]
                        else:
                            inb, outb = set(), set()
                        ft, hit = [(1, 0) if gw in (wc1, wc2, fh) else (min(max(ft - len(inb), 0)+1, 2) , (len(inb)>ft)*4*(len(inb)-ft))][0]
                        team = evolve_team_no_price_info(team, inb, outb)
            

                    df = pd.DataFrame.from_dict(datapoints, orient='index')
                    df['season'], df['rank'] = season, rank
                
                    # save 
                    SAVED_DF = pd.concat([SAVED_DF, df])
                    ALREADY_DONE.add((season, rank))
                    #print('Completed person in ', round(time.time() - start) , ' seconds')


                X.append(df.drop(['did_wc', 'season', 'rank'], axis=1))
                y.append(df['did_wc'])
                rank += 1
                failed_in_a_row = 0
            except Exception as e:
                print('FAILED')
                print(e)
                rank += 1
                failed_in_a_row += 1

    # save 
    SAVED_DF.reset_index(drop=True).to_csv(SAVED_WILDCARD_TRAINING_PATH)

    X = pd.concat(X).reset_index(drop=True)
    y = pd.concat(y).reset_index(drop=True)

    # CROSSVALIDATION

    # CREATE NEURAL NETWORK
    model, feature_names = train_rf_with_threshold(X,y,125,.008, crossval= True, num_rounds=3, metric='mse')
    model_name = DROPBOX_PATH + 'models/' + 'wildcard_copying' + "/" + f'season{season}_n{n_per_season}' + '.sav' 
    try:
        os.makedirs(DROPBOX_PATH + 'models/' + 'wildcard_copying')
    except:
        pass
    save_model((model, feature_names), model_name)



# print out the predictions for a few people to see some good thresholds
def test_classifier_with_top_players(classifier, season, wildcard_end_season, ranks, league='Overall', league_n=250,when_transfer='late'):
    ''' REMOVE THIS JUST TEMPORARY '''"""
    model_name = DROPBOX_PATH + 'models/' + 'wildcard_copying' + "/" + f'season{season}_n{5}' + '.sav' 
    actual = datapoint['did_wc']
    prediction = model.predict([[datapoint[x] for x in feature_names]])
    print(f'gw{gw}  _|  Guess: {prediction}  --  Actual: {actual}')

    """
    BENCH_FACTOR = .15
    if when_transfer == 'late':
        field_models, keeper_models = ['field_all_batch'], ['keeper_all_batch']
    elif when_transfer == 'early':
        field_models, keeper_models = ['field_early_transfer_batch'], ['keeper_all_batch']
    model, feature_names = load_model(classifier)
    data_df = get_data_df(20, season)
    name_df = data_df[['element', 'name']].drop_duplicates()
    fix_df = pd.read_csv(DROPBOX_PATH + f"20{str(season)[:2]}-{str(season)[2:]}/fix_df.csv", index_col=0)
    

    print('season: ', season)
    all_rank_datapoints = {}
    for rank in ranks:
        meta_df, gwks_df = get_meta_gwks_dfs(season, league, league_n, rank)
        wc1, wc2, fh, bb, tc = [int(x) if not(math.isnan(x)) else 39 for x in meta_df[['wildcard1', 'wildcard2', 'free_hit', 'bench_boost', 'triple_captain']].to_numpy()[0]]
        if wc1 == 39:
            wc1 = wildcard_end_season - 1
            print(f'rank{rank}: manual wc1')
        if wc2 == 39:
            wc2 = 38
            print(f'rank{rank}: manual wc2')

        points_each_week = get_league_player_weekly_scores(season, [rank], league = league, interval = league_n)[rank]

        team = set(meta_df[[f'player_{i}' for i in range(1,16)]].to_numpy()[0])
        datapoints = {}
        ft = 0
        for gw in range(1,39): #only weeks where option to play
            #print(f'gw{gw}')

            ## CREATE DATAPOINT
            if  gw in list(range(3, wc1+1)) + list(range(wildcard_end_season, wc2+1)): #only weeks where option to play
                gw_tm = top_player_create_transfer_market(data_df, name_df, field_models, keeper_models, gw)
                chip_played_dict = {'freehit':fh, 'triple_captain':tc, 'bench_boost':bb}
                bench = set(gwks_df.loc[gwks_df['gw']==gw][[f'bench{i}' for i in range(1,5)]].to_numpy()[0])
                datapoint = create_wildcard_datapoint(data_df.loc[data_df['gw']==gw], fix_df, bench, points_each_week, gw_tm, wildcard_end_season, BENCH_FACTOR, chip_played_dict, team, ft, gw)
                prediction = model.predict([[datapoint[x] for x in feature_names]])[0]
                datapoints[gw] = prediction

                actual = ('yes' if gw in (wc1, wc2) else 'no')
                print(f'gw{gw}  _|  Guess: {prediction}  --  Actual: {actual}')

            ## EVOLVE TEAM
            if gw != 1 and gw != fh:
                inb, outb = [eval(x) for x in gwks_df.loc[gwks_df['gw']==gw][['inb', 'outb']].to_numpy()[0]]
            else:
                inb, outb = set(), set()
            ft, hit = [(1, 0) if gw in (wc1, wc2, fh) else (min(max(ft - len(inb), 0)+1, 2) , (len(inb)>ft)*4*(len(inb)-ft))][0]   
            team = evolve_team_no_price_info(team, inb, outb)

        all_rank_datapoints[rank] = (wc1, wc2), datapoints

    return all_rank_datapoints


#train_wildcard_predictor([2021], [17], 5)
#train_wildcard_predictor([2021], [17], 100)
#train_wildcard_predictor([2021], [17], 1000)
#train_wildcard_predictor([2021], [17], 2500)
#train_wildcard_predictor([2021], [17], 3000)
#train_wildcard_predictor([2021], [17], 4000)
#train_wildcard_predictor([2021], [17], 5000)
#train_wildcard_predictor([2021], [17], 6000)
#train_wildcard_predictor([2021], [17], 8000)
#train_wildcard_predictor([2021], [17], 9000)
#train_wildcard_predictor([2021], [17], 1000, no_gw=True)
N = 4000
SEASON = 2021
model_name = DROPBOX_PATH + 'models/' + 'wildcard_copying' + "/" + f'season{SEASON}_n{N}' + '.sav' 
all_ranks = test_classifier_with_top_players(model_name, SEASON, 17,range(4000, 4005))#,  league= 'White Hart Kane', league_n= 7)
for rank, info in all_ranks.items():
    wcs, datapoints = info
    print(f"Rank{rank} - wc1={wcs[0]}  wc2={wcs[1]}")
    print(f"Top 4 Weeks = {[f'gw{gw}: {score}   ' for gw, score in sorted(datapoints.items(), key=lambda x: x[1], reverse=True)[:4]]}")
    print(f"{datapoints}\n")
    
