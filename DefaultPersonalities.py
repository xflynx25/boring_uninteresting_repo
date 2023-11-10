# needs season and folder
from constants import DROPBOX_PATH
FIELD_MODELS = ['full', 'full_squared_error','onehot', 'priceless', 'no_ict',\
    'no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info']
KEEPER_MODELS = ['keeper_engineering', 'keeper_engineering_squared_error',\
    'keeper_extra_crossval','keeper_no_price_mse']

default_p =  {
    'login_credentials' : ['email', 'password', 'id'],
    'allowed_healths' : ['a'], #list - i.e.['a','d']
    'max_hit' : 4, #int, positive
    'bench_factors' : (0.15, 0.015),
    'value_vector' : [0.15, 0.25, 0.6],# ['a', 'd']],#[.20,.15,.65], #listlike - [worth, next_match_delta, full_match_delta]
    'num_options' : 10, #int - how many to consider in transfer search single n
    'quality_factor' : 3, #float - multiplier for relative quality of rank-n top transfer
    'hesitancy_dict' : {
        1: {0: .4, 1: .4, 2:.8},
        2: {1: .4, 2: .4, 3:.8}
    }, #dict - keys:ft values=dict with keys=num_transfers, vals = 0-1 float
    'min_delta_dict': {
        1: {0: 0, 1: .25, 2: 1.25},
        2: {0: 0, 1: 0, 2: .75, 3: 1.75}
    },
    'earliest_chip_weeks': {'wildcard':(9, 22), 'freehit': 22, 'triple_captain': 22, 'bench_boost':22},#{'wildcard':6, 'freehit': 21, 'triple_captain': 21, 'bench_boost':21}, # dict with keys= str of chip name, vals are int-gw (wildcard is tuple)
    'chip_threshold_construction': {
        'wildcard': [.555,1.75,'avg'], 'freehit': [1,2,'avg'], 'triple_captain': [1,2,'avg'], 'bench_boost':[1,2,'avg']
    }, #{chip: (%of top, std_deviations, choice_method)} #where choice method is either 'max', 'min', 'avg'
    # suggested ranges (0-2, 0-4, oneof('max','min','avg'))
    'chip_threshold_tailoffs' : [.3,.19,.19,.19], #float - 0-1, .2 tails off around gw34, .1 around gw30, .05 around gw20, slow tailoff
    'wildcard_method': 'modern', #'classical',# (modern is by using modelling of top players where classical just treats it like another chip)
    'wildcard_model_path':  DROPBOX_PATH + r"models/Current/wildcard_copying/season2021_n5000.sav",
    'player_protection': 0,
    'field_model_suites': [FIELD_MODELS],
    'keeper_model_suites': [KEEPER_MODELS],
    'bad_players': [],#['mahrez', 'Pog', 'wErNe', 'ramsdale']
    'nerf_info': [],#[('ederson', 5/6)]
    'force_remake': False, #if not using the same modelling as personality who runs before you, force_remake should be true
    'when_transfer': 'late',
    'early_transfer_aggressiveness': 3, #higher is greater
    'early_transfer_evolution_length': 20, #number is gw when revert to normal
}

default_a = {
    #### likely change
    'allowed_healths' : ['a'], #list - i.e.['a','d']
    'bad_players': [],#['mahrez', 'Pog', 'wErNe', 'ramsdale']
    'bench_factors' : (.075,.0075),
    'value_vector' : [0.1, 0.5, 0.4],# ['a', 'd']], #listlike - [worth, next_match_delta, full_match_delta]
    'earliest_chip_weeks': {'wildcard':(9, 22), 'freehit': 19, 'triple_captain': 19, 'bench_boost':19},#{'wildcard':6, 'freehit': 21, 'triple_captain': 21, 'bench_boost':21}, # dict with keys= str of chip name, vals are int-gw (wildcard is tuple)
    'max_hit' : 8, #int, positive, for now we know we can't afford 4 transfers 
    'min_delta_dict': {
        1: {0: 0, 1: .25, 2: 1, 3:1.75},
        2: {0: 0, 1: 0, 2: .75, 3: 1.5, 4:2.25}
    },
    'hesitancy_dict' : {
        1: {0: 0.4, 1: 0.3, 2: 0.6, 3: 0.7},
        2: {1: 0.4, 2: 0.3, 3: 0.6, 4: 0.7}
    }, #dict - keys:ft values=dict with keys=num_transfers, vals = 0-1 float 
    
    #### maybe change 
    'num_options' : 12, #int - how many to consider in transfer search single n
    'quality_factor' : 4.5, #float - multiplier for relative quality of rank-n top transfer
    'chip_threshold_construction': {
        'wildcard': [.555,1.75,'avg'], 'freehit': [1,2,'avg'], 'triple_captain': [1,2,'avg'], 'bench_boost':[1,2,'avg']
    }, #{chip: (%of top, std_deviations, choice_method)} #where choice method is either 'max', 'min', 'avg'
    # suggested ranges (0-2, 0-4, oneof('max','min','avg'))
    'chip_threshold_tailoffs' : [.3,.2, .2, .2], #float - 0-1, .2 tails off around gw34, .1 around gw30, .05 around gw20, slow tailoff
    'player_protection': 0,
    'early_transfer_aggressiveness': 4, #higher is greater
    'early_transfer_evolution_length': 20, #number is gw when revert to normal
    
    #### rarely change
    'wildcard_method': 'modern', #'classical',# (modern is by using modelling of top players where classical just treats it like another chip)
    'wildcard_model_path':  DROPBOX_PATH + r"models/Current/wildcard_copying/season2021_n8000.sav", 
    'nerf_info': [],#[('ederson', 5/6)]
    'when_transfer': 'late',
    'field_model_suites': [FIELD_MODELS],
    'keeper_model_suites': [KEEPER_MODELS],
    'force_remake': False, #if not using the same modelling as personality who runs before you, force_remake should be true
    'login_credentials' : ['email', 'password', 'id'],
}

DEFAULT_PERSONALITY_LOOKUP = {
    'Athena-v1.2a': default_a, 
    'Athena-v1.2p': default_p, 
}
DEFAULT_PERSONALITY_NAMES = list(DEFAULT_PERSONALITY_LOOKUP.keys())

def create_default_personality(season, folder, personality_key):
    return {**{
    'folder': folder, 
    'season': season
    }, **DEFAULT_PERSONALITY_LOOKUP[personality_key]}