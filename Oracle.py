### FUNCTIONS ### 
"""
DATABASE WORK ~~
train_model_suite

MAKING TRANSFER MARKET ~~
make_transfer_market
this_week_predictions
make_model_dict

POST PROCESSING OF TRANSFER MARKET ~~
adjust_team_player_prices
adjust_team_expected_pts_full
adjust_blank_gameweek_pts_N1
make_team_players
change_team_players
avg_transfer_markets
full_transfer_creation: handle_field_players & handle_keepers

"""

from general_helpers import get_columns_containing, safe_make_folder, safer_eval
import pandas as pd
import time
import math
from private_versions.constants import DROPBOX_PATH, UPDATED_TRAINING_DB, TRANSFER_MARKET_SAVED,  MAX_FORWARD_CREATED
from Oracle_helpers import save_model, load_model, drop_back_and_forward_features, train_rf_with_threshold,\
    drop_nan_rows, get_backward, long_term_benchwarmer, get_sets_from_name, blank_row, is_useful_model,\
    keeper_outfield_split, eliminate_players, nerf_players, save_market, visualize_top_transfer_market
from os import chdir, listdir


"""
### GETTING THE MODELS ####
"""

# Create Suite of Model-Type
# df is TrainingDf but with possibly some players rid of per specific model
def train_model_suite(training_db, folder, names, n, threshold, crossval=False, metric='mse', n_starter_cols=None):
    print("Folder= ", folder,"\n")
    safe_make_folder( DROPBOX_PATH + 'models/' + folder + "/")
    for name in names:
        print("model= ", name,"\n")
        start = time.time()

        back, forward, target = get_sets_from_name(name)
        df = training_db.copy()
    
        # rid of no minutes in 6 weeks
        if 6 in back:
            df = df.loc[df['minutes_L6']>0]

        # rid of blanks for 1wk predictions
        if target == 1:
            df = df.loc[df['FIX1_num_opponents']>0]

        # rid of gameweeks for model forward/back
        df = df.loc[(df['gw']>max(back)) & (df['gw']<40-target)]

        # get the y column
        targ = 'total_points_N' + str(target)
        target_df = df[targ]

        # rid of features for model forward/back
        df = drop_back_and_forward_features(df, back, forward, target)
        
        #get the matrices for regression & train model
        X,y = drop_nan_rows(df, target_df)
        print('training model')
        model, feature_names = train_rf_with_threshold(X,y,n,threshold, crossval=crossval, metric=metric, n_starter_cols=n_starter_cols)
        model_name = DROPBOX_PATH + 'models/' + folder + "/" + name + '.sav' 
        save_model((model, feature_names), model_name)

        end = time.time()
        print('took ', (end-start)/60, " minutes.")


"""
### MAKING TRANSFER MARKET / USING THE MODELS ####
"""

# Make Transfer Market (model-type)
#@param: df with processed players, int gw
#@return: transfer market df with (element, position, team, value, expected pts N1, expected pts full)
def make_transfer_market(processed_players, gw, player_healths, model_suite, preloaded=False, model_folder = 'Current'):
    df = processed_players.loc[processed_players['gw']==gw].reset_index(drop=True)
    # Make Model Dict
    model_dict = make_model_dict(gw, model_suite, preloaded=preloaded, model_folder = model_folder)
    # Player Prediction
    single = this_week_predictions(df, model_dict, gw, 1)
    full = this_week_predictions(df, model_dict, gw, 'full')
    meta = df[['element','position','team','value']]
    new_cols = pd.merge(single, full, how='left', left_on=['element'], right_on = ['element']) #pd.concat([single, full], axis=1)
    prehealth = pd.merge(meta, new_cols, how='left', left_on=['element'], right_on = ['element']) #pd.concat([meta, new_cols], axis=1).reset_index(drop=True)
    final = pd.merge(prehealth, player_healths, how='left', left_on=['element'], right_on = ['element'])
    final.to_csv(DROPBOX_PATH + "weekly_scoreboard.csv")
    return final

# @params: df loc'ed at this gw, dictionary map names to ML model, length= 'full' or 1
    # model_dict: keys are (backward, forward)
# @return: a series which is the predicted points for each player in the list, with proper title
# @Note: will predict 0 for any player after gw 34 who has not been in the league for at least 6 games for full.
# @Note: predicts 0 for players benched 6 games in a row, or if blank gw for this gw
def this_week_predictions(this_week_player_stats, model_dict, gw, length):
    gw = int(gw)
    if length == 'full':
        forward = (6 if gw < 34 else 39-gw)
    elif length == 1:
        forward = 1
    else:
        raise Exception("Only supported prediction lengths are 1 and 'full'")

    backward_ops = sorted(list(set([x[0] for x in model_dict if x[1] == forward])), reverse=True) #sorted descending

    predictions = []
    for _, row in this_week_player_stats.iterrows():
        
        player_id = row['element']
        row = row.to_frame().T
        backward = get_backward(row, backward_ops, forward) #need to add forward to clear out FIX6 if less time remaining
        key = backward, forward
        
        #Either automatic 0 or we use model to make prediction
        if key not in model_dict or (backward==6 and long_term_benchwarmer(row)) or (forward==1 and blank_row(row)):
            #print("Cause of auto-zero: ", key, ':for key: ',  key not in model_dict, (backward==6 and long_term_benchwarmer(row)), (forward==1 and blank_row(row)))
            predictions.append([player_id, 0])
        else:
            
            model, feature_names = model_dict[key]
            prediction = model.predict(row[feature_names]) #preserves order of the columns
            predictions.append([player_id, prediction[0]])

    name = ('expected_pts_N1' if length == 1 else 'expected_pts_full')
    preds = pd.DataFrame(predictions, columns=['element', name])
    return preds
 
        
#@param: int current gw
#@return: model dictionary name to model for possible models we might want to use 
# overloading this for use by evaluator helpers, if preloaded=False
def make_model_dict(gw, model_suite, preloaded=False, model_folder = 'Current'): 
    chdir(DROPBOX_PATH + f"models/{model_folder}/" + model_suite)
    files = listdir()

    model_dict = {}
    for filename in files:
        back, _, target = get_sets_from_name(filename[:-4])
        key = (max(back), target)
        if is_useful_model(key, gw):
            if preloaded: #OVERLOADING
                model, feature_names = preloaded[model_suite][filename]
            else:
                model, feature_names  = load_model(filename)
            model_dict[key] = (model, feature_names)
        
    return model_dict



"""
### POST PROCESSING OF TRANSFER MARKET ####
"""


#@params: full_transfer_market = df with 'element','position','team','value', 'expected_pts_N1', 'expected_pts_full'
#           squad is list of (element, sell_value)
#@return: replace values with the sell-values of the players
def adjust_team_player_prices(full_transfer_market, squad):
    for element, sell_value in squad:
        indx = full_transfer_market.loc[full_transfer_market['element']==element].index[0]
        full_transfer_market.at[indx, 'value'] = sell_value 
    return full_transfer_market


def adjust_team_expected_pts_full(full_transfer_market, player_injury_penalties):
    for element, penalty in player_injury_penalties.items():
        this_player = full_transfer_market.loc[full_transfer_market['element']==element]
        predicted_pts = this_player['expected_pts_full'].to_list()[0]
        indx = this_player.index[0]
        full_transfer_market.at[indx, 'expected_pts_full'] = penalty * predicted_pts
    return full_transfer_market

def adjust_blank_gameweek_pts_N1(full_transfer_market, blank_players):
    for player in blank_players:
        indx = full_transfer_market.loc[full_transfer_market['element']==player].index[0]
        full_transfer_market.at[indx, 'expected_pts_N1'] = 0  #previously .at
    return full_transfer_market

# same params as above but full_transfer_market should be updated
# returns df with only the elements in the team (should be size 15, 6)
def make_team_players(full_transfer_market, squad):
    player_ids = [x[0] for x in squad]
    team_players = full_transfer_market.loc[full_transfer_market['element'].isin(player_ids)]
    return team_players 

#@params: 15,6 team players df, transfers= df with('set(inb)', 'set(outb)', delta), will just be empty if save_ft
#@return: 15,6 df but with some swaps based on transfers
def change_team_players(full_transfer_market, team_players, transfers):
    print('Trnasfers in oracle: \n', transfers)
    inbound = safer_eval(transfers['inbound'])
    outbound = safer_eval(transfers['outbound'])
    new_boys = full_transfer_market.loc[full_transfer_market['element'].isin(inbound)]
    team_players = team_players.loc[~team_players['element'].isin(outbound)]
    new_team_players = pd.concat([team_players, new_boys],axis=0).reset_index(drop=True)
    return new_team_players

#@param: list of transfer markets ('element','position','team','value', 'expected_pts_N1', 'expected_pts_full')
#@return: expected points columns averaged from the markets
# save gives the current_gw number if we're saving
def avg_transfer_markets(transfer_markets, name_df=None, visualize=False, save=False, suite_names=[], model_folder='Current'):
    base_info = transfer_markets[0][['element','position','team','value','status']].sort_values('element').reset_index(drop=True)

    point_columns = []
    market_index = 0
    for df in transfer_markets:
        if save:
            market = suite_names[market_index]
            market_index += 1
            path = DROPBOX_PATH + f"models/{model_folder}/" + market + '/transfer_market_history.csv'
            save_market(save, df, path)

        df = df.sort_values('element').reset_index(drop=True)
        df = df[['expected_pts_N1', 'expected_pts_full']]
        point_columns.append(df)


    averages = pd.concat([each.stack() for each in point_columns],axis=1)\
            .apply(lambda x:x.mean(),axis=1)\
            .unstack()

    final = pd.concat([base_info, averages], axis=1)

    if visualize:
        for df, suitename in zip(transfer_markets + [final], suite_names + ['final']):
            print(f"\n\nNew Transfer Market Type is {suitename}")
            for score_type in ('expected_pts_N1', 'expected_pts_full'):
                print("\nSorted by ", score_type, "\nName   N1     N6")
                top_perf = df.sort_values(score_type, ascending=False)[:25]
                for _, player in top_perf.iterrows():
                    element, n1, full = player[['element', 'expected_pts_N1', 'expected_pts_full']].to_list()
                    name = name_df.loc[name_df['element']==element]['name'].tolist()[0]

                    """begoviC wasn't printing"""
                    try:
                        print(name, "  ", n1, "  ", full)
                    except:
                        printable_name = ""
                        for letter in name:
                            try:
                                print(letter)
                                printable_name += letter
                            except:
                                printable_name += '_'
                        print(printable_name, "  ", n1, "  ", full)
                
    return final


#have to deal with individual positions, dgw stuff, and early/late
# ASSUMPTION: dgw models imply their no_dgw counterpart also has been trained
# save gives the current_gw number if we're saving
def handle_field_players(field_suites, outfield, health_outfield, gw, name_df, visualize=False, save=False, preloaded=False, model_folder = 'Current'):
    individuals = False
    sparse_individuals = False
    dgwks = []
    field_models = []
    ordered_field_suites = []
    for suite in field_suites:
        if 'early' in suite and gw >= 20:
            pass 
        elif 'late' in suite and gw < 20:
            pass
        elif 'dgw' == suite:
            dgwks.append(suite)
        elif suite == 'individuals':
            individuals = True
        elif suite == 'sparse_individuals':
            sparse_individuals = True
        else:
            field_models.append(make_transfer_market(outfield, gw, health_outfield, suite, preloaded=preloaded, model_folder = model_folder))
            ordered_field_suites.append(suite) #track suites

    if individuals:
        #print('doing individuals, successful code')    
        individual_field_models = []
        suite_key = {2: 'defenders', 3: 'midfielders', 4: 'forwards'}
        for i in (2,3,4):
            suite = suite_key[i]
            df_positional = outfield.loc[outfield['position']==float(i)]

            positional_regr = make_transfer_market(df_positional, gw, health_outfield, suite, preloaded=preloaded, model_folder = model_folder)
            individual_field_models.append(positional_regr)

        players_ind_regressions = pd.concat(individual_field_models, axis=0)
        field_models.append(players_ind_regressions)
        ordered_field_suites.append('individuals') #track suites

    if sparse_individuals:
        #print('doing sparse individuals, very successful code')    
        individual_field_models = []
        suite_key = {2: 'defenders_sparse', 3: 'midfielders_sparse', 4: 'forwards_sparse'}
        for i in (2,3,4):
            suite = suite_key[i]
            df_positional = outfield.loc[outfield['position']==float(i)]
            positional_regr = make_transfer_market(df_positional, gw, health_outfield, suite, preloaded=preloaded, model_folder = model_folder)
            individual_field_models.append(positional_regr)

        players_sparse_ind_regressions = pd.concat(individual_field_models, axis=0)
        field_models.append(players_sparse_ind_regressions)
        ordered_field_suites.append('sparse_individuals') #track suites
            
    if dgwks:
        # dgw_upcoming will predict for the single week but we throw it away
        print('successfully doing dgwks yeah')
        this_n = 0
        num_opp_columns, num_players_outfield = get_columns_containing(['_num_opponents'], outfield).columns, outfield.shape[0]
        for n in range(MAX_FORWARD_CREATED, 0, -1):
            this_col = f'FIX{n}_num_opponents'
            if this_col in num_opp_columns:
                if outfield[this_col].isna().sum() < num_players_outfield:
                    this_n = n
                    break
        if this_n == 0:
            raise Exception('Says there are no valid num_opponents things to choose from')
        
        yes_upcoming = outfield.loc[outfield[f'FIX{this_n}_num_opponents']> this_n]
        no_upcoming = outfield.loc[outfield[f'FIX{this_n}_num_opponents']<= this_n]
        yes_this = outfield.loc[outfield['FIX1_num_opponents']>1]
        no_this = outfield.loc[outfield['FIX1_num_opponents']<=1]

        these = []
        base_info = ['element','position','team','value','status']
        for option, score_type, yes, no in (['dgw', '_N1', yes_this, no_this], ['dgw_upcoming', '_full', yes_upcoming, no_upcoming]):
            anti_option = 'no_' + option
            score_col = 'expected_pts' + score_type
            if yes.shape[0] == 0:
                total = make_transfer_market(no, gw, health_outfield, anti_option, preloaded=preloaded, model_folder = model_folder)
            elif no.shape[0] == 0:
                total = make_transfer_market(yes, gw, health_outfield, option, preloaded=preloaded, model_folder = model_folder)
            else:
                yes_df = make_transfer_market(yes, gw, health_outfield, option, preloaded=preloaded, model_folder = model_folder)
                no_df = make_transfer_market(no, gw, health_outfield, anti_option, preloaded=preloaded, model_folder = model_folder)
                total = pd.concat([yes_df, no_df], axis=0)
            these.append(total[base_info + [score_col]])
            
        merged_dgw = pd.merge(left=these[0], right=these[1], how='outer', on=base_info)
        #print(merged_dgw.shape, '   ', merged_dgw.shape[0]*2 , ' vs ', get_columns_containing(['expected_pts'], merged_dgw.astype(bool).sum(axis=0)).sum())
        ordered_field_suites.append('dgw')
        field_models.append(merged_dgw)
            
    if len(field_models) == 0: #no working models for this week
        print('WARNING: no working models this week gw', gw)
        return pd.DataFrame()
    dffieldplayers = avg_transfer_markets(field_models, name_df, visualize=visualize, save=save, suite_names=ordered_field_suites, model_folder = model_folder)
    return dffieldplayers

# save gives the current_gw number if we're saving
def handle_keepers(keeper_suites, keepers, health_keepers, gw, name_df, visualize=False,save=False, preloaded=False, model_folder = 'Current'):
    keeper_models = []
    for suite in keeper_suites:
        keeper_models.append(make_transfer_market(keepers, gw, health_keepers, suite, preloaded=preloaded, model_folder = model_folder))
    if len(keeper_models) == 0: #no working models for this week
        return pd.DataFrame()
    dfkeepers = avg_transfer_markets(keeper_models, name_df, visualize=visualize, save=save, suite_names=keeper_suites, model_folder = model_folder)
    return dfkeepers

# @param: stats df, health df, models, info about anti-preferences, options
# @return: the requested df with expected points for all the players
def full_transfer_creation(current_gw_stats, health_df, field_suites, keeper_suites, bad_players, nerf_info,\
    adjustment_information, name_df=None, visualize=False, force_remake=False, save=False, model_folder='Current'):

    full_transfer_market = pd.read_csv(TRANSFER_MARKET_SAVED, index_col=0)
    if full_transfer_market.shape[0]==0 or force_remake: #if we want to remake the base transfer market
        gw = current_gw_stats['gw'].to_list()[0] 
        keepers, health_keepers, outfield, health_outfield = keeper_outfield_split(current_gw_stats, health_df)
        
        ### NEW MODELS FOR FIELD PLAYERS ###
        dffieldplayers = handle_field_players(field_suites, outfield, health_outfield, gw, name_df, visualize=visualize, save=save, model_folder=model_folder)

        ### NEW MODELS FOR KEEPERS ###
        dfkeepers = handle_keepers(keeper_suites, keepers, health_keepers, gw, name_df, visualize=visualize, save=save, model_folder=model_folder)

        ### COMBINING THE TWO ###
        full_transfer_market = pd.concat([dfkeepers, dffieldplayers],axis=0)
        full_transfer_market.to_csv(TRANSFER_MARKET_SAVED)

    '''Nerfing and Eliminating Players'''
    nerf_elements, nerf_scales = [x[0] for x in nerf_info], [x[1] for x in nerf_info]
    full_transfer_market = nerf_players(full_transfer_market, nerf_elements, name_df, nerf_scales, visualize=False)
    full_transfer_market = eliminate_players(full_transfer_market, bad_players, name_df, visualize=visualize).reset_index(drop=True)
    
    '''adjusting player prices in transfer market, making pd team representation'''
    squad, player_injury_penalties, blank_players = adjustment_information 
    price_adjusted_transfer_market = adjust_team_player_prices(full_transfer_market, squad)
    point_adjusted_transfer_market = adjust_team_expected_pts_full(price_adjusted_transfer_market, player_injury_penalties)
    full_transfer_market = adjust_blank_gameweek_pts_N1(point_adjusted_transfer_market, blank_players)
    return full_transfer_market

                      
#@param: requires the points_each_week (only needed for the previous weeks)
   # chip_play_weeks: is 39 if hasn't been played yet otherwise the play week (for current usage)
# creates a datapoint for wildcard prediction for a given gameweek 
def create_wildcard_datapoint(current_gw_df, fix_df, bench, points_each_week, gw_tm, wildcard_end, bench_factor, chip_play_weeks, team_list, ft, gw):
    datapoint = {}
    datapoint['ft'] = ft
    chip_play_weeks_ordered = [chip_play_weeks[x] for x in ('freehit', 'bench_boost', 'triple_captain')]
    datapoint['fh_used'], datapoint['bb_used'], datapoint['tc_used'] = [play_week if (gw >= play_week and not(math.isnan(play_week))) else 0 for play_week in chip_play_weeks_ordered]
    past_pts = list(points_each_week.values())[:gw-1]
    datapoint['gw'], datapoint['wks_till_expire'] = gw, (wildcard_end-gw if wildcard_end > gw else 38 - gw)
    datapoint['pts_season'], datapoint['pts_last_1'], datapoint['pts_last_3'], datapoint['pts_last_6'] =\
        [sum(past_pts[-x:]) for x in (len(past_pts), 1, 3, 6)]
    print(gw, fix_df)
    datapoint['week_interval'] = fix_df.loc[fix_df['gw']==gw]['day'].to_list()[0] - max(fix_df.loc[fix_df['gw']<gw]['day'].to_list())
    for i in (1,3,6):
        double_teams = {team: sum([fix_df.loc[(fix_df['gw']==some_gw) & (fix_df['team']==team)].shape[0] > 1 for some_gw in list(range(gw, gw+i))])  for team in fix_df['team'].unique()}
        blank_teams =  {team: sum([fix_df.loc[(fix_df['gw']==some_gw) & (fix_df['team']==team)].shape[0] == 0.0 for some_gw in list(range(gw, gw+i))])  for team in fix_df['team'].unique()}
        datapoint[f'ngames_all_N{i}'] = fix_df.loc[fix_df['gw'].isin(list(range(gw, gw+i)))].shape[0]/2
        datapoint[f'nblank_all_N{i}'], datapoint[f'ndgw_all_N{i}'] = [sum(l.values()) for l in (blank_teams, double_teams)]
        datapoint[f'nblank_team_N{i}'], datapoint[f'ndgw_team_N{i}'] = [sum([l[int(current_gw_df.loc[current_gw_df['element']==elem]['team'].max())] for elem in team_list]) for l in (blank_teams, double_teams) ]
    
    team_player_data_gw = current_gw_df.loc[current_gw_df['element'].isin(team_list)]
    for n in(1,3):
        player_minutes = team_player_data_gw[f'minutes_L{n}'].to_list()
        num_reds = sum(team_player_data_gw[f'red_cards_L{n}'].to_list())
        datapoint[f'last{n}_red_cards'] = (num_reds if not(math.isnan(num_reds)) else 0)
        for i in (1, 60, 90):
            datapoint[f'last{n}_avged_less_than{i}_minutes'] = len([x for x in player_minutes if x < n*i])
            
    datapoint['pred_pts_next_1'], datapoint['pred_pts_next_6'] = [gw_tm.loc[gw_tm['element'].isin(team_list)][f'expected_pts_{when}'].sum() -\
        (1-bench_factor) * gw_tm.loc[gw_tm['element'].isin(bench)][f'expected_pts_{when}'].sum() for when in ('N1', 'full')]

    return datapoint

