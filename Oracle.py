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

from random import randint
import pandas as pd
import numpy as np
import time
from constants import DROPBOX_PATH, UPDATED_TRAINING_DB, TRANSFER_MARKET_SAVED
from Oracle_helpers import save_model, load_model, drop_back_and_forward_features, train_rf_with_threshold,\
    drop_nan_rows, get_backward, long_term_benchwarmer, get_sets_from_name, blank_row, is_useful_model,\
    keeper_outfield_split, eliminate_players, nerf_players, save_market
from os import chdir, listdir


"""
### GETTING THE MODELS ####
"""

# Create Suite of Model-Type
# df is TrainingDf but with possibly some players rid of per specific model
def train_model_suite(training_db, folder, names, n, threshold, crossval=False, metric='mae'):
    print("Folder= ", folder,"\n")
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
        model, feature_names = train_rf_with_threshold(X,y,n,threshold, crossval=crossval, metric=metric)
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
def make_transfer_market(processed_players, gw, player_healths, model_suite):
    df = processed_players.loc[processed_players['gw']==gw].reset_index(drop=True)
    # Make Model Dict
    model_dict = make_model_dict(gw, model_suite)

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
# @return: a series which is the predicted points for each player in the list, with proper title
# @Note: will predict 0 for any player after gw 34 who has not been in the league for at least 6 games for full.
# @Note: predicts 0 for players benched 6 games in a row, or if blank gw for this gw
def this_week_predictions(this_week_player_stats, model_dict, gw, length):
    backward_ops = sorted(list(set([x[0] for x in model_dict])), reverse=True) #sorted descending

    if length == 'full':
        forward = [6 if gw < 34 else 1][0]#39-gw][0] 
    elif length == 1:
        forward = 1
    else:
        raise Exception("Only supported prediction lengths are 1 and 'full'")

    predictions = []
    for _, row in this_week_player_stats.iterrows():
        
        player_id = row['element']
        row = row.to_frame().T
        backward = get_backward(row, backward_ops, forward) #need to add forward to clear out FIX6 if less time remaining
        key = backward, forward
        
        #Either automatic 0 or we use model to make prediction
        if key not in model_dict or (backward==6 and long_term_benchwarmer(row)) or (forward==1 and blank_row(row)):
            predictions.append([player_id, 0])
        else:
            
            model, feature_names = model_dict[key]
            prediction = model.predict(row[feature_names]) #preserves order of the columns
            predictions.append([player_id, prediction[0]])

    name = ['expected_pts_N1' if length == 1 else 'expected_pts_full'][0]
    preds = pd.DataFrame(predictions, columns=['element', name])
    return preds

        
#@param: int current gw
#@return: model dictionary name to model for possible models we might want to use 
def make_model_dict(gw, model_suite): 
    chdir(DROPBOX_PATH + "/models/" + model_suite)
    files = listdir()

    model_dict = {}
    for filename in files:
        back, _, target = get_sets_from_name(filename[:-4])
        key = (max(back), target)
        if is_useful_model(key, gw):
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
        predicted_pts = this_player['expected_pts_full']
        indx = this_player.index[0]
        full_transfer_market.at[indx, 'expected_pts_full'] = penalty * predicted_pts
    return full_transfer_market

def adjust_blank_gameweek_pts_N1(full_transfer_market, blank_players):
    for player in blank_players:
        indx = full_transfer_market.loc[full_transfer_market['element']==player].index[0]
        full_transfer_market.at[indx, 'expected_pts_N1'] = 0 
    return full_transfer_market

# same params as above but full_transfer_market should be updated
# returns df with only the elements in the team (should be size 15, 6)
def make_team_players(full_transfer_market, squad):
    player_ids = [x[0] for x in squad]
    team_players = full_transfer_market.loc[full_transfer_market['element'].isin(player_ids)]
    return team_players 

#@params: 15,6 team players df, transfers= df with([set(inb)], [set(outb)], delta), will just be empty if save_ft
#@return: 15,6 df but with some swaps based on transfers
def change_team_players(full_transfer_market, team_players, transfers):
    inbound = transfers['inbound'][0]
    outbound = transfers['outbound'][0]
    new_boys = full_transfer_market.loc[full_transfer_market['element'].isin(inbound)]
    team_players = team_players.loc[~team_players['element'].isin(outbound)]
    new_team_players = pd.concat([team_players, new_boys],axis=0).reset_index(drop=True)
    return new_team_players

#@param: list of transfer markets ('element','position','team','value', 'expected_pts_N1', 'expected_pts_full')
#@return: expected points columns averaged from the markets
# save gives the current_gw number if we're saving
def avg_transfer_markets(transfer_markets, name_df=None, visualize=False, save=False, suite_names=[]):
    base_info = transfer_markets[0][['element','position','team','value','status']].sort_values('element').reset_index(drop=True)

    point_columns = []
    market_index = 0
    for df in transfer_markets:
        if save:
            market = suite_names[market_index]
            market_index += 1
            path = DROPBOX_PATH + "/models/" + market + 'transfer_market_history.csv'
            save_market(save, df, path)

        df = df.sort_values('element').reset_index(drop=True)
        df = df[['expected_pts_N1', 'expected_pts_full']]
        point_columns.append(df)


    averages = pd.concat([each.stack() for each in point_columns],axis=1)\
            .apply(lambda x:x.mean(),axis=1)\
            .unstack()

    final = pd.concat([base_info, averages], axis=1)

    if visualize:
        for df in transfer_markets + [final]:
            print("\n\nNew Transfer Market Type")
            for score_type in ('expected_pts_N1', 'expected_pts_full'):
                print("\nSorted by ", score_type, "\nName   N1     N6")
                top_perf = df.sort_values(score_type, ascending=False)[:25]
                for _, player in top_perf.iterrows():
                    element, n1, full = player[['element', 'expected_pts_N1', 'expected_pts_full']].to_list()
                    name = name_df.loc[name_df['element']==element]['name'].tolist()[0]
                    print(name, "  ", n1, "  ", full)
                
    return final


#have to deal with individual positions, dgw stuff, and early/late
# save gives the current_gw number if we're saving
def handle_field_players(field_suites, outfield, health_outfield, gw, name_df, visualize=False, save=False):
    individuals = False
    sparse_individuals = False
    no_dgwks = []
    dgwks = []
    field_models = []
    ordered_field_suites = []
    for suite in field_suites:
        if 'early' in suite and gw >= 20:
            pass 
        elif 'late' in suite and gw < 20:
            pass
        elif 'no_dgw' in suite:
            no_dgwks.append(suite)
        elif 'dgw' in suite:
            dgwks.append(suite)
        elif suite == 'individuals':
            individuals = True
        elif suite == 'sparse_individuals':
            sparse_individuals = True
        else:
            field_models.append(make_transfer_market(outfield, gw, health_outfield, suite))
            ordered_field_suites.append(suite) #track suites

    if individuals:
        print('doing individuals, successful code')    
        individual_field_models = []
        suite_key = {2: 'defenders', 3: 'midfielders', 4: 'forwards'}
        for i in (2,3,4):
            suite = suite_key[i]
            df_positional = outfield.loc[outfield['position']==float(i)]

            positional_regr = make_transfer_market(df_positional, gw, health_outfield, suite)
            individual_field_models.append(positional_regr)

        players_ind_regressions = pd.concat(individual_field_models, axis=0)
        field_models.append(players_ind_regressions)
        ordered_field_suites.append('individuals') #track suites

    if sparse_individuals:
        print('doing sparse individuals, very successful code')    
        individual_field_models = []
        suite_key = {2: 'defenders_sparse', 3: 'midfielders_sparse', 4: 'forwards_sparse'}
        for i in (2,3,4):
            suite = suite_key[i]
            df_positional = outfield.loc[outfield['position']==float(i)]
            positional_regr = make_transfer_market(df_positional, gw, health_outfield, suite)
            individual_field_models.append(positional_regr)

        players_sparse_ind_regressions = pd.concat(individual_field_models, axis=0)
        field_models.append(players_sparse_ind_regressions)
        ordered_field_suites.append('sparse_individuals') #track suites
            
    for option in dgwks:
        anti_option = 'no_' + option
        if anti_option in no_dgwks:
            print('successfully doing dgwks yeah')
            if 'upcoming' in option:
                yes = outfield.loc[outfield['FIX6_num_opponents']>6]
                no = outfield.loc[outfield['FIX6_num_opponents']<=6]
            else:
                yes = outfield.loc[outfield['FIX1_num_opponents']>1]
                no = outfield.loc[outfield['FIX1_num_opponents']<=1]

            if yes.shape[0] == 0:
                field_models.append(make_transfer_market(no, gw, health_outfield, anti_option))
                ordered_field_suites.append(anti_option) #track suites
            elif no.shape[0] == 0:
                field_models.append(make_transfer_market(yes, gw, health_outfield, option))
                ordered_field_suites.append(option) #track suites
            else:
                yes_df = make_transfer_market(yes, gw, health_outfield, option)
                no_df = make_transfer_market(no, gw, health_outfield, anti_option)
                total = pd.concat([yes_df, no_df], axis=0)
                field_models.append(total)
                ordered_field_suites.append(option) #track suites, just save into the positive dgw one for both

    dffieldplayers = avg_transfer_markets(field_models, name_df, visualize=visualize, save=save, suite_names=ordered_field_suites)
    #correct for how I did not put a model in for predicting full for the single gw predict for is dgw
    if dgwks.count('dgw') == 1:
        new_expected_pts_full = dffieldplayers.apply(lambda x: x['expected_pts_full']*len(field_models) / (len(field_models)-1), axis=1)
        new_expected_pts_full.name = 'expected_pts_full'
        dffieldplayers = dffieldplayers.drop('expected_pts_full', axis=1)
        dffieldplayers = pd.concat([dffieldplayers, new_expected_pts_full], axis=1)
    return dffieldplayers

# save gives the current_gw number if we're saving
def handle_keepers(keeper_suites, keepers, health_keepers, gw, name_df, visualize=False,save=False):
    keeper_models = []
    for suite in keeper_suites:
        keeper_models.append(make_transfer_market(keepers, gw, health_keepers, suite))
    dfkeepers = avg_transfer_markets(keeper_models, name_df, visualize=visualize, save=save, suite_names=keeper_suites)
    return dfkeepers

# @param: stats df, health df, models, info about anti-preferences, options
# @return: the requested df with expected points for all the players
def full_transfer_creation(current_gw_stats, health_df, field_suites, keeper_suites, bad_players, nerf_info,\
    adjustment_information, name_df=None, visualize=False, force_remake=False, save=False):

    full_transfer_market = pd.read_csv(TRANSFER_MARKET_SAVED, index_col=0)
    if full_transfer_market.shape[0]==0 or force_remake: #if we want to remake the base transfer market
        gw = current_gw_stats['gw'].to_list()[0] 
        keepers, health_keepers, outfield, health_outfield = keeper_outfield_split(current_gw_stats, health_df)
        
        ### NEW MODELS FOR FIELD PLAYERS ###
        dffieldplayers = handle_field_players(field_suites, outfield, health_outfield, gw, name_df, visualize=visualize, save=save)

        ### NEW MODELS FOR KEEPERS ###
        dfkeepers = handle_keepers(keeper_suites, keepers, health_keepers, gw, name_df, visualize=visualize, save=save)

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


    