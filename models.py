""" 
Here lies the code for creating the df, feature pair that constitutes a model suite.
It will also return the folder name.
These models are called by user manually in create many model suites
"""
#%%
from constants import UPDATED_TRAINING_DB, DROPBOX_PATH
from Oracle import train_model_suite
from Oracle_helpers import load_model
from general_helpers import drop_columns_containing, get_columns_containing
import time
import pandas as pd 
TRAIN_DF = pd.read_csv(UPDATED_TRAINING_DB, index_col=0)
TRAIN_DF = drop_columns_containing(['element', 'season', 'name'],TRAIN_DF)
TRAIN_DF = TRAIN_DF.drop('team', axis=1)

KEEPER_RAW_DF = TRAIN_DF.loc[TRAIN_DF['position']==1.0]
KEEPER_DF = drop_columns_containing(['creativity', 'ict', 'threat', 'transfers', 'selected'],KEEPER_RAW_DF)
FIELD_DF = TRAIN_DF.loc[TRAIN_DF['position']!=1.0]

FULL_MODEL_NAMES = ['1236-136-6', '123-136-6','1-136-6','1-1-1','123-1-1',\
    '1236-1-1']#,'1236-135-5', '1236-124-4', '1236-123-3', '1236-12-2']
    

CROSSVAL = True
THRESHOLD = [.0001, .0004,.0008, .0011, .0015, .002 , .003, .004, .005, .006,\
    .007, .0085, .01, .012, .015, .02, .025, .03, .035,.04,.045, .05, .06,.07,.08,.09,.1, .15, .2, .3, .4]
TREE_SIZE = 175 

# model returns folder and df with only columns/rows we would like to use in suite
# suite takes care of applicable to all like beg/end of year, blanks, long-bench, 

# just one hot
def one_hot_only(folder):
    df = drop_columns_containing(['pts_goal', 'pts_cs', 'position'], FIELD_DF)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

def manual_keeper_engineering(folder):
    df = drop_columns_containing(['pts_goal', 'pts_cs', 'position'], KEEPER_DF)
    opp_cols = get_columns_containing(['FIX'], df)
    opp_cols = drop_columns_containing(['a_'], opp_cols)    
    team_cols = drop_columns_containing(['FIX'], df)
    team_cols = get_columns_containing(['a_'], team_cols) 
    rest_cols = ['gw','minutes', 'value', 'ppmin', 'day', 'odds','goals', 'saves','pts','Ff_',\
        'clean_sheets','bps','bonus','concessions', 'cards','num', 'influence','total_points']
    all_columns = opp_cols.columns.to_list() + team_cols.columns.to_list() + rest_cols
    df = get_columns_containing(all_columns, df)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

def keeper_extra_crossval(folder):
    df = drop_columns_containing(['pts_goal', 'pts_cs', 'position'], KEEPER_DF)
    opp_cols = get_columns_containing(['FIX'], df)
    opp_cols = drop_columns_containing(['a_'], opp_cols)    
    team_cols = drop_columns_containing(['FIX'], df)
    team_cols = get_columns_containing(['a_'], team_cols) 
    rest_cols = ['gw','minutes', 'value', 'ppmin', 'day', 'odds','goals', 'saves','pts','Ff_',\
        'clean_sheets','bps','bonus','concessions', 'cards','num', 'influence','total_points']
    all_columns = opp_cols.columns.to_list() + team_cols.columns.to_list() + rest_cols
    df = get_columns_containing(all_columns, df)
    threshold = []
    for i in range(len(THRESHOLD)): #adding in the means between each number to go slower 
        threshold.append(THRESHOLD[i])
        if i != len(THRESHOLD) - 1:
            threshold.append(round(sum(THRESHOLD[i:i+2])/2,6))
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, threshold, crossval=CROSSVAL*2)

def manual_keeper_engineering_squared_error(folder):
    df = drop_columns_containing(['pts_goal', 'pts_cs', 'position'], KEEPER_DF)
    opp_cols = get_columns_containing(['FIX'], df)
    opp_cols = drop_columns_containing(['a_'], opp_cols)    
    team_cols = drop_columns_containing(['FIX'], df)
    team_cols = get_columns_containing(['a_'], team_cols) 
    rest_cols = ['gw','minutes', 'value', 'ppmin', 'day', 'odds','goals', 'saves','pts','Ff_',\
        'clean_sheets','bps','bonus','concessions', 'cards','num', 'influence','total_points']
    all_columns = opp_cols.columns.to_list() + team_cols.columns.to_list() + rest_cols
    df = get_columns_containing(all_columns, df)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL, metric='mse')

# all the representations
def full_positional_representation(folder):
    df = FIELD_DF
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

def full_positional_representation_squared_error(folder):
    df = FIELD_DF
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL, metric='mse')

# all positions individually
def individual_position(position, folder):
    df = TRAIN_DF.loc[TRAIN_DF['position']==float(position)]
    df = drop_columns_containing(['pts_goal', 'pts_cs', 'position', 'is_pos'], df)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

# all positions individually, no main indicators
def individual_position_sparse(position, folder):
    df = TRAIN_DF.loc[TRAIN_DF['position']==float(position)]
    df = drop_columns_containing(['pts_goal', 'pts_cs', 'position', 'is_pos'], df)
    df = drop_columns_containing(['transfers','selected', 'ict_index', 'value', 'creativity','threat','influence'], df)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

# no price
def no_price(folder):
    df = drop_columns_containing(['value'], FIELD_DF)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

def keepers_no_price_mse(folder):
    df = drop_columns_containing(['pts_goal', 'pts_cs', 'position'], KEEPER_DF)
    opp_cols = get_columns_containing(['FIX'], df)
    opp_cols = drop_columns_containing(['a_'], opp_cols)    
    team_cols = drop_columns_containing(['FIX'], df)
    team_cols = get_columns_containing(['a_'], team_cols) 
    rest_cols = ['gw','minutes', 'ppmin', 'day', 'odds','goals', 'saves','pts','Ff_',\
        'clean_sheets','bps','bonus','concessions', 'cards','num', 'influence','total_points']
    all_columns = opp_cols.columns.to_list() + team_cols.columns.to_list() + rest_cols
    df = get_columns_containing(all_columns, df)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL, metric='mse')


# vary by season phase
def first_half_season(folder):
    df = FIELD_DF.loc[FIELD_DF['gw']<20]
    model_names = FULL_MODEL_NAMES[:6]
    train_model_suite(df, folder, model_names, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

def second_half_season(folder):
    df = FIELD_DF.loc[FIELD_DF['gw']>=20]
    model_names = FULL_MODEL_NAMES
    train_model_suite(df, folder, model_names, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)


# double gw
def double_gw(folder):
    df = FIELD_DF.loc[FIELD_DF['FIX1_num_opponents']==2]
    model_names = FULL_MODEL_NAMES[3:6]
    train_model_suite(df, folder, model_names, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

# > 6 games in next 6
def double_gw_upcoming(folder):
    df = FIELD_DF.loc[FIELD_DF['FIX6_num_opponents']>6]
    model_names = FULL_MODEL_NAMES
    train_model_suite(df, folder, model_names, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)


def no_double_gw(folder):
    df = FIELD_DF.loc[FIELD_DF['FIX1_num_opponents']<2]
    model_names = FULL_MODEL_NAMES[3:6]
    train_model_suite(df, folder, model_names, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

def no_double_gw_upcoming(folder):
    df = FIELD_DF.loc[FIELD_DF['FIX6_num_opponents']<=6]
    model_names = FULL_MODEL_NAMES
    train_model_suite(df, folder, model_names, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

# no ict --> to see what will rise when the most important is removed
def no_ict_index(folder):
    df = drop_columns_containing(['ict_index'], FIELD_DF)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

#this is most volatile year to year and not even sure if i did it correctly
def no_transfer_info(folder):
    df = drop_columns_containing(['transfers','selected'], FIELD_DF)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

#try without these dominating features
def no_ict_transfers_price(folder):
    df = drop_columns_containing(['transfers','selected', 'ict_index', 'value'], FIELD_DF)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

def no_ictany_transfers_price(folder):
    df = drop_columns_containing(['transfers','selected', 'ict_index', 'value', 'creativity','threat','influence'], FIELD_DF)
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, crossval=CROSSVAL)

##### VISUALIZATION #####
def print_model_features(folder, name, top_n='max'):
    
    path = DROPBOX_PATH + 'models/' + folder + '/' + name + '.sav'
    model, feature_names = load_model(path)

    associations = [x for x in zip(model.feature_importances_, feature_names)]
    top_n = [len(associations) if top_n=='max' else top_n][0]
    associations.sort(key=lambda x: x[0], reverse=True)[:top_n]
    print("num features: ", len(associations))

    for imp, name in associations:
        print(name, round(10**6*imp)/10**6)

''' TRAINING THE MODEL SUITES '''
#%%  For My Computer
#folder_names = ['keeper_engineering', 'keeper_engineering_squared_error','keeper_extra_crossval','full']
#models = [manual_keeper_engineering,manual_keeper_engineering_squared_error,keeper_extra_crossval, full_positional_representation]
folder_names = ['no_transfer_info', 'full_squared_error','onehot', 'no_ict_transfers_price','no_dgw', 'no_dgw_upcoming']
models = [no_transfer_info,full_positional_representation_squared_error, one_hot_only, no_ict_transfers_price, no_double_gw, no_double_gw_upcoming]


for folder, model in zip(folder_names, models):
    print('starting')
    start = time.time()
    model(folder)
    end = time.time() 
    print("\n", folder, "took ", (end-start)/60, " minutes\n\n")
# %% For Loaner Computer
"""
folder_names = ['defenders', 'midfielders', 'forwards', 'priceless', 'no_ict', 'dgw', 'early', 'late', 'dgw_upcoming']
models = [
    lambda x: individual_position(2, x), lambda x: individual_position(3, x), lambda x: individual_position(4, x),\
        no_price, no_ict_index, double_gw, first_half_season, second_half_season, double_gw_upcoming]
"""
"""
folder_names = ['no_ictANY_transfers_price']
models = [no_ictany_transfers_price]
"""
"""
folder_names = ['keeper_no_price_mse']
models = [keepers_no_price_mse]
"""
folder_names = ['defenders_sparse', 'midfielders_sparse', 'forwards_sparse']
models = [
    lambda x: individual_position_sparse(2, x), lambda x: individual_position_sparse(3, x), lambda x: individual_position_sparse(4, x)]

for folder, model in zip(folder_names, models):
    print('starting')
    start = time.time()
    model(folder)
    end = time.time() 
    print("\n", folder, "took ", round((end-start)/60), " minutes\n\n")
# %%
