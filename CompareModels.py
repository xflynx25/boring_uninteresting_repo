'''
Call this file with different model folders, or combination of models, 
    and see what it is predicting for the week. 
'''

import pandas as pd
import Oracle
from constants import TRANSFER_MARKET_SAVED, DROPBOX_PATH
import Accountant
from general_helpers import safe_read_csv
from Agent import injury_penalties, get_current_gw

FIELD_MODELS = ['full', 'full_squared_error','onehot', 'priceless', 'no_ict',\
    'no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info']
KEEPER_MODELS = ['keeper_engineering', 'keeper_engineering_squared_error',\
    'keeper_extra_crossval','keeper_no_price_mse']
    
FIELD_MODELS_EARLY = ['no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info', 'no_ict',\
    'no_ict_transfers_price']


gw = get_current_gw()
name_df = Accountant.make_name_df()
health_df = Accountant.make_and_save_health_df(gw)
current_gw_stats = safe_read_csv(DROPBOX_PATH + "current_stats.csv") #speedup if multiple personalities

# once we fill this out it should work
squad = [] # - HACKK
player_injury_penalties = injury_penalties(gw, health_df, [x[0] for x in squad])#just the elements
blank_players = current_gw_stats.loc[current_gw_stats['FIX1_num_opponents']==0]['element'] #blank players get 0    
adjustment_information = squad, player_injury_penalties, blank_players 

#####################
""" YOU EDIT ZONE """
#####################
model_lists = {
    'new': [FIELD_MODELS, KEEPER_MODELS],
    'new':[FIELD_MODELS_EARLY, KEEPER_MODELS],
    'Current': [FIELD_MODELS, KEEPER_MODELS],
    'Current':[FIELD_MODELS_EARLY, KEEPER_MODELS],
}

#########################
""" END YOU EDIT ZONE """
#########################


for i, (model_folder, (field_models, keeper_models)) in enumerate(model_lists.items()):
    
    print(f'\nLooking at combo # {i}\n\n')
    full_transfer_market = Oracle.full_transfer_creation(current_gw_stats, health_df, field_models, keeper_models, [],\
        [], [0], name_df=name_df, visualize=True, force_remake=True, save=False, model_folder = model_folder)#gw) #claim don't need to save because we can just recompute with evaluator at the end of the season 

    #full_transfer_market = pd.read_csv(TRANSFER_MARKET_SAVED, index_col=0)
    Oracle.visualize_top_transfer_market(full_transfer_market, name_df, 'expected_pts_N1', 35, healthy=health_df, allowed_healths=['a','d']) 
    Oracle.visualize_top_transfer_market(full_transfer_market, name_df, 'expected_pts_full', 35, healthy=health_df, allowed_healths=['a','d'])
