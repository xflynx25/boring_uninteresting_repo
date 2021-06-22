from constants import DROPBOX_PATH
FIELD_MODELS = ['full', 'full_squared_error','onehot','early', 'late','priceless',\
    'no_ict', 'no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info',\
    'dgw',  'no_dgw', 'individuals', 'sparse_individuals'] #'dgw_upcoming','no_dgw_upcoming',
    
KEEPER_MODELS = ['keeper_engineering', 'keeper_engineering_squared_error',\
    'keeper_extra_crossval','keeper_no_price_mse']

personality1 = {
        'season' : '2020-21',
        'login_credentials' : ('_________', '__________', ''' INTEGER '''), #EMAIL, PASSWORD, TEAM ID
        'folder' : DROPBOX_PATH + "___________/", #bot_folder_name
        'allowed_healths' : ['a', 'd'], #list - i.e.['a','d']
        'max_hit' : 8, #int, positive
        'bench_factors' : (.2,.025),
        'value_vector' : [.15,.25,.6], #listlike - [worth, next_match_delta, full_match_delta]
        'num_options' : 10, #int - how many to consider in transfer search single n
        'quality_factor' : 2, #float - multiplier for relative quality of rank-n top transfer
        'hesitancy_dict' : {
            1: {0: .6, 1: .3, 2: .6, 3:.7, 4:.6, 5:.6, 6:.7},
            2: {1: .4, 2: .25, 3: .5, 4:.6, 5:.6, 6:.7, 7:.8}
        }, #dict - keys:ft values=dict with keys=num_transfers, vals = 0-1 float
        'min_delta_dict': {
            1: {0: 0, 1: .25, 2: 1, 3:2},
            2: {0: 0, 1: 0, 2: .75, 3: 1.5, 4:2.5, 5:3, 6:4, 7:5}
        },
        'earliest_chip_weeks' : {'wildcard':28, 'freehit': 28, 'triple_captain': 28, 'bench_boost':28},#{'wildcard':6, 'freehit': 21, 'triple_captain': 21, 'bench_boost':21}, # dict with keys= str of chip name, vals are int-gw (wildcard is tuple)
        'chip_threshold_construction': {
            'wildcard': [1,1.75,'avg'], 'freehit': [1,1.75,'avg'], 'triple_captain': [1,1.75,'avg'], 'bench_boost':[1,1.75,'avg']
        }, #{chip: (%of top, std_deviations, choice_method)} #where choice method is either 'max', 'min', 'avg'
        # suggested ranges (0-2, 0-4, oneof('max','min','avg'))
        'chip_threshold_tailoff' : .19, #float - 0-1, .2 tails off around gw34, .1 around gw30, .05 around gw20, slow tailoff
        'player_protection': 0, #int, you will not sell any player in the top __ predicted for these next few weeks.
        'field_model_suites':FIELD_MODELS,
        'keeper_model_suites': KEEPER_MODELS, 
        'bad_players': ['ramsdale', 'de gea'],#['mahrez', 'Pog', 'wErNe', 'ramsdale']
        'nerf_info': [],#[('ederson', 5/6)]
        'force_remake': False, #if not using the same modelling as personality who runs before you, force_remake should be true
}

Athena_v10a = {
        'season' : '2020-21',
        'login_credentials' : ('_________', '__________', ''' INTEGER '''), #EMAIL, PASSWORD, TEAM ID
        'folder' : DROPBOX_PATH + "___________/", #bot_folder_name
        'allowed_healths' : ['a'], #list - i.e.['a','d']
        'max_hit' : 8, #int, positive, for now we know we can't afford 4 transfers 
        'bench_factors' : (.15,.0075),
        'value_vector' : [.15,.35,.5,],# ['a', 'd']], #listlike - [worth, next_match_delta, full_match_delta]
        'num_options' : 15, #int - how many to consider in transfer search single n
        'quality_factor' : 3.5, #float - multiplier for relative quality of rank-n top transfer
        'hesitancy_dict' : {
            1: {0: .5, 1: .4, 2: .6, 3:.7},
            2: {1: .4, 2: .3, 3: .4, 4:.6}  #CURRENTLY ALTERING VERY FREQUENTLY
        }, #dict - keys:ft values=dict with keys=num_transfers, vals = 0-1 float 
        'min_delta_dict': {
            1: {0: 0, 1: .25, 2: 1, 3:1.75},
            2: {0: 0, 1: 0, 2: .75, 3: 1.5, 4:2.25}
        },
        'earliest_chip_weeks' : {'wildcard':30, 'freehit': 30, 'triple_captain': 30, 'bench_boost':30},#{'wildcard':6, 'freehit': 21, 'triple_captain': 21, 'bench_boost':21}, # dict with keys= str of chip name, vals are int-gw (wildcard is tuple)
        'chip_threshold_construction': {
            'wildcard': [1,1.5,'avg'], 'freehit': [1,1.5,'avg'], 'triple_captain': [1,1.5,'avg'], 'bench_boost':[1,1.5,'avg']
        }, #{chip: (%of top, std_deviations, choice_method)} #where choice method is either 'max', 'min', 'avg'
        # suggested ranges (0-2, 0-4, oneof('max','min','avg'))
        'chip_threshold_tailoff' : .19, #float - 0-1, .2 tails off around gw34, .1 around gw30, .05 around gw20, slow tailoff
        'player_protection': 0,
        'field_model_suites':FIELD_MODELS,
        'keeper_model_suites': KEEPER_MODELS,
        'bad_players': ['de gea'],#['mahrez', 'Pog', 'wErNe', 'ramsdale']
        'nerf_info': [],#[('ederson', 5/6)]
        'force_remake': False, #if not using the same modelling as personality who runs before you, force_remake should be true
}

Athena_v10p = {
        'season' : '2020-21',
        'login_credentials' : ('_________', '__________', ''' INTEGER '''), #EMAIL, PASSWORD, TEAM ID
        'folder' : DROPBOX_PATH + "___________/", #bot_folder_name
        'allowed_healths' : ['a'], #list - i.e.['a','d']
        'max_hit' : 0, #int, positive
        'bench_factors' : (.25,.05),
        'value_vector' : [.25, .3, .45,],# ['a', 'd']],#[.20,.15,.65], #listlike - [worth, next_match_delta, full_match_delta]
        'num_options' : 8, #int - how many to consider in transfer search single n
        'quality_factor' : 2, #float - multiplier for relative quality of rank-n top transfer
        'hesitancy_dict' : {
            1: {0: .5, 1: .5},
            2: {1: .5, 2: .5}
        }, #dict - keys:ft values=dict with keys=num_transfers, vals = 0-1 float
        'min_delta_dict': {
            1: {0: 0, 1: .25},
            2: {0: 0, 1: 0, 2: .75}
        },
        'earliest_chip_weeks': {'wildcard':30, 'freehit': 30, 'triple_captain': 30, 'bench_boost':30},#{'wildcard':6, 'freehit': 21, 'triple_captain': 21, 'bench_boost':21}, # dict with keys= str of chip name, vals are int-gw (wildcard is tuple)
        'chip_threshold_construction': {
            'wildcard': [1,1.5,'avg'], 'freehit': [1,1.5,'avg'], 'triple_captain': [1,1.5,'avg'], 'bench_boost':[1,1.5,'avg']
        }, #{chip: (%of top, std_deviations, choice_method)} #where choice method is either 'max', 'min', 'avg'
        # suggested ranges (0-2, 0-4, oneof('max','min','avg'))
        'chip_threshold_tailoff' : .19, #float - 0-1, .2 tails off around gw34, .1 around gw30, .05 around gw20, slow tailoff
        'player_protection': 0,
        'field_model_suites': FIELD_MODELS,
        'keeper_model_suites': KEEPER_MODELS,
        'bad_players': ['de gea'],#['mahrez', 'Pog', 'wErNe', 'ramsdale']
        'nerf_info': [],#[('ederson', 5/6)]
        'force_remake': False, #if not using the same modelling as personality who runs before you, force_remake should be true
}

personalities_to_run = [personality1, Athena_v10a, Athena_v10p]  
#personalities_to_run = [Athena_v10a]
#personalities_to_run = [Athena_v10p]