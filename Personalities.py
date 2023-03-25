from DefaultPersonalities import create_default_personality
from constants import DROPBOX_PATH
FIELD_MODELS = ['full', 'full_squared_error','onehot','early', 'late','priceless',\
    'no_ict', 'no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info',\
    'dgw', 'individuals', 'sparse_individuals'] #'dgw_upcoming'

"""TEMP WHILE THE OTHER MODELS ARE TRAINING UP"""
FIELD_MODELS = ['full', 'full_squared_error','onehot', 'priceless', 'no_ict',\
    'no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info']

KEEPER_MODELS = ['keeper_engineering', 'keeper_engineering_squared_error',\
    'keeper_extra_crossval','keeper_no_price_mse']

FIELD_MODELS_EARLY = ['no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info', 'no_ict',\
    'no_ict_transfers_price']


mercury = {
        'season' : '2022-23',
        'login_credentials' : ('vaflynnsbot1@gmail.com', 'Alphafpl2022!', 6150359),
        'folder' : DROPBOX_PATH + "HelloFPL1.2/", #str filepath
        'allowed_healths' : ['a'], #list - i.e.['a','d']
        'max_hit' : 8, #int, positive, for now we know we can't afford 4 transfers 
        'bench_factors' : (.075,.0075),
        'value_vector' : [0.15, 0.35, 0.5],# ['a', 'd']], #listlike - [worth, next_match_delta, full_match_delta]
        'num_options' : 12, #int - how many to consider in transfer search single n
        'quality_factor' : 4.5, #float - multiplier for relative quality of rank-n top transfer
        'hesitancy_dict' : { #.3 --> ~2.07  & .7 --> ~ 0.5 & .8 --> ~ .25 & .2 --> ~4 & .6 like .7
            1: {0: 0.4, 1: 0.3, 2: 0.6, 3: 0.7},
            2: {1: 0.4, 2: 0.3, 3: 0.6, 4: 0.7}
        }, #dict - keys:ft values=dict with keys=num_transfers, vals = 0-1 float 
        'min_delta_dict': {
            1: {0: 0, 1: .25, 2: 1, 3:1.75},
            2: {0: 0, 1: 0, 2: .75, 3: 1.5, 4:2.25}
        },
        'earliest_chip_weeks': {'wildcard':(9, 18), 'freehit': 19, 'triple_captain': 19, 'bench_boost':19},#{'wildcard':6, 'freehit': 21, 'triple_captain': 21, 'bench_boost':21}, # dict with keys= str of chip name, vals are int-gw (wildcard is tuple)
        'chip_threshold_construction': {
            'wildcard': [.555,1.75,'avg'], 'freehit': [1,2,'avg'], 'triple_captain': [1,2,'avg'], 'bench_boost':[1,2,'avg']
        }, #{chip: (%of top, std_deviations, choice_method)} #where choice method is either 'max', 'min', 'avg'
        # suggested ranges (0-2, 0-4, oneof('max','min','avg'))
        'chip_threshold_tailoffs' : [.3,.19,.19,.19], #float - 0-1, .2 tails off around gw34, .1 around gw30, .05 around gw20, slow tailoff
        'wildcard_method': 'modern', #'classical',# (modern is by using modelling of top players where classical just treats it like another chip)
        'wildcard_model_path':  DROPBOX_PATH + r"models/Current/wildcard_copying/season2021_n8000.sav", 
        'player_protection': 0,
        'field_model_suites': [FIELD_MODELS],
        'keeper_model_suites': [KEEPER_MODELS],
        'bad_players': [],#['mahrez', 'Pog', 'wErNe', 'ramsdale']
        'nerf_info': [],#[('ederson', 5/6)]
        'force_remake': False, #if not using the same modelling as personality who runs before you, force_remake should be true
        'when_transfer': 'late',
        'early_transfer_aggressiveness': 4, #higher is greater
        'early_transfer_evolution_length': 20, #number is gw when revert to normal
}
# 'wildcard': [1,1.5,'avg'], 'freehit': [1,1.5,'avg'], 'triple_captain': [1,1.5,'avg'], 'bench_boost':[1,1.5,'avg']



Athena_v10a = {
        'season' : '2022-23',
        'login_credentials' : ('athenav1.0a@gmail.com', 'Alphafpl2022!', 6140897),
        'folder' : DROPBOX_PATH + "Athena-v1.2a/", #str filepath
        'allowed_healths' : ['a'], #list - i.e.['a','d']
        'max_hit' : 8, #int, positive, for now we know we can't afford 4 transfers 
        'bench_factors' : (.075,.0075),
        'value_vector' : [0.1, 0.5, 0.4],# ['a', 'd']], #listlike - [worth, next_match_delta, full_match_delta]
        'num_options' : 12, #int - how many to consider in transfer search single n
        'quality_factor' : 4.5, #float - multiplier for relative quality of rank-n top transfer
        'hesitancy_dict' : {
            1: {0: 0.4, 1: 0.3, 2: 0.6, 3: 0.7},
            2: {1: 0.4, 2: 0.3, 3: 0.6, 4: 0.7}
        }, #dict - keys:ft values=dict with keys=num_transfers, vals = 0-1 float 
        'min_delta_dict': {
            1: {0: 0, 1: .25, 2: 1, 3:1.75},
            2: {0: 0, 1: 0, 2: .75, 3: 1.5, 4:2.25}
        },
        'earliest_chip_weeks': {'wildcard':(9, 22), 'freehit': 19, 'triple_captain': 19, 'bench_boost':19},#{'wildcard':6, 'freehit': 21, 'triple_captain': 21, 'bench_boost':21}, # dict with keys= str of chip name, vals are int-gw (wildcard is tuple)
        'chip_threshold_construction': {
            'wildcard': [.555,1.75,'avg'], 'freehit': [1,2,'avg'], 'triple_captain': [1,2,'avg'], 'bench_boost':[1,2,'avg']
        }, #{chip: (%of top, std_deviations, choice_method)} #where choice method is either 'max', 'min', 'avg'
        # suggested ranges (0-2, 0-4, oneof('max','min','avg'))
        'chip_threshold_tailoffs' : [.3,.2, .2, .2], #float - 0-1, .2 tails off around gw34, .1 around gw30, .05 around gw20, slow tailoff
        'wildcard_method': 'modern', #'classical',# (modern is by using modelling of top players where classical just treats it like another chip)
        'wildcard_model_path':  DROPBOX_PATH + r"models/Current/wildcard_copying/season2021_n8000.sav", 
        'player_protection': 0,
        'field_model_suites': [FIELD_MODELS],
        'keeper_model_suites': [KEEPER_MODELS],
        'bad_players': [],#['mahrez', 'Pog', 'wErNe', 'ramsdale']
        'nerf_info': [],#[('ederson', 5/6)]
        'force_remake': False, #if not using the same modelling as personality who runs before you, force_remake should be true
        'when_transfer': 'late',
        'early_transfer_aggressiveness': 4, #higher is greater
        'early_transfer_evolution_length': 20, #number is gw when revert to normal
}
# 'wildcard': [1,1.5,'avg'], 'freehit': [1,1.5,'avg'], 'triple_captain': [1,1.5,'avg'], 'bench_boost':[1,1.5,'avg']


Athena_v10p = {
        'season' : '2022-23',
        'login_credentials' : ('athenav1.0p@gmail.com', 'Alphafpl2022!', 6144050),
        'folder' : DROPBOX_PATH + "Athena-v1.2p/", #str filepath
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

Athena_v10a22 = create_default_personality('2022-23', DROPBOX_PATH + "Athena-v1.2gw22/", "Athena-v1.2p")
Athena_v10a23 = create_default_personality('2022-23', DROPBOX_PATH + "Athena-v1.2gw23/", "Athena-v1.2p")

#personalities_to_run = [mercury]
personalities_to_run = [Athena_v10p, Athena_v10a]
personalities_to_run = [Athena_v10p, Athena_v10a, Athena_v10a22, Athena_v10a23]
#personalities_to_run = [Athena_v10a23]
