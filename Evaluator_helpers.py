from constants import DROPBOX_PATH, TM_FOLDER_ROOT
from general_helpers import get_data_df, get_meta_gwks_dfs, get_counts
import pandas as pd
import random
import math 
import os
CENTURY = 20

FIELD_MODELS = ['full', 'dgw', 'full_squared_error','onehot','early', 'late','priceless',\
    'no_ict', 'no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info',\
    'individuals', 'sparse_individuals']
    
KEEPER_MODELS = ['keeper_engineering', 'keeper_engineering_squared_error',\
    'keeper_extra_crossval','keeper_no_price_mse']

FIELD_MODELS_EARLY = ['no_ict_transfers_price', 'no_ictANY_transfers_price','no_transfer_info']

MIXED_MODELS = [['field_early_transfer_batch'], ['field_all_batch']], [['keeper_all_batch'], ['keeper_all_batch']] # this what we want to test for mixed

MALLEABLE_PARAMS = ['max_hit','bench_factors','value_vector','num_options','quality_factor','hesitancy_dict','min_delta_dict',\
    'earliest_chip_weeks','chip_threshold_construction','chip_threshold_tailoff','player_protection',\
    'field_model_suites','keeper_model_suites','when_transfer']

def get_athena_skeleton(season):
    return {
    'season' : f'{CENTURY}{str(season)[:2]}-{str(season)[2:4]}',
    'login_credentials' : ('email', 'password', 11223344),
    'folder' : DROPBOX_PATH + "Simulation/athena_Simulation_instance/", #str filepath
    'allowed_healths' : ['a'], #list - i.e.['a','d']
    'bad_players': [],
    'nerf_info': [],
    'force_remake': False, #if not using the same modelling as personality who runs before you, force_remake should be true
    'wildcard_method': 'modern', #'classical',# (modern is by using modelling of top players where classical just treats it like another chip)
    'wildcard_model_path':  DROPBOX_PATH + "models/Current/wildcard_copying/season2021_n5000.sav", 
         
    #  DEFAULTS  #
    'max_hit': 8,
    'bench_factors': (.15,.015),
    'value_vector': [.15,.35,.5],
    'num_options': 10, 
    'quality_factor': 3, 
    'hesitancy_dict': {
            1: {0: .4, 1: .3, 2: .5, 3:.6},# 4:.7},
            2: {1: .4, 2: .3, 3: .5, 4:.6}#, 5:.7}  
        },
    'min_delta_dict': {
            1: {0: 0, 1: .25, 2: 1, 3: 1.75},# 4: 2.5},
            2: {0: 0, 1: 0, 2: .75, 3: 1.5, 4:2.25}#, 5: 3.0}
        },
    'earliest_chip_weeks': {'wildcard':(4, 15), 'freehit': 15, 'triple_captain': 15, 'bench_boost': 15},
    'chip_threshold_construction': {
            'wildcard': [.555,1.75,'avg'], 'freehit': [1,2,'avg'], 'triple_captain': [1,2,'avg'], 'bench_boost':[1,2,'avg']
        },
    'chip_threshold_tailoffs' : [.3,.15,.15,.15],
    'player_protection': 0,
    'field_model_suites': [['field_all_batch']], #enclosed for the different season stages as referenced in when_transfer 
    'keeper_model_suites': [['keeper_all_batch']],
    'when_transfer': 'late', #'early', [list(range(1,19)), list(range(19, 39))]
    }

def this_week_chip(gw, meta_df):
    chip_names = ['wildcard1','wildcard2','bench_boost','triple_captain','free_hit']
    chip_truths = (meta_df[chip_names] == gw).to_numpy()[0]
    played_chips = [chip_names[i] for i in range(len(chip_truths)) if chip_truths[i]]
    chip = ['none' if len(played_chips) == 0 else played_chips[0]][0]
    return chip

# gives the new team if team is computing price_data
def evolve_team(team, inbound, outbound, price_data):
    if len(inbound):
        new_players = pd.DataFrame(index=range(len(inbound)),columns=['element','position', 'purchase','now','sell'])
        old_players = team.loc[~team['element'].isin(outbound)]
        price_data_new = price_data.loc[price_data['element'].isin(inbound)]
        fail = False
        if price_data_new.shape[0] != len(inbound):
            print(price_data_new, inbound, outbound, 'this was the price data failing')
            for _, row in price_data.iterrows():
                print(row[['element', 'value', 'team']].to_list())
            fail = True
        new_players.loc[:, ['element', 'position', 'purchase','sell']] = price_data_new[['element', 'position','value','value']].to_numpy()
        team = pd.concat([old_players, new_players],axis=0).reset_index(drop=True)
        if team.shape[0] != 15 or fail:
            print('New Team Does not have 15 unique players: \n', team['element'].to_list(), ' ', inbound,' ',  outbound)
            print('Old players and New players\n', old_players, '\n',new_players)
    return team


# gives the new team if team is not computing price_data
def evolve_team_no_price_info(team, inb, outb):
    team = team.union(inb)
    team = team.difference(outb)
    return team
# @param: season is '2020-21' format
def id_to_teamname_converter_saved(season):
    df = pd.read_csv(DROPBOX_PATH + f'Our_Datasets/{season}/team_converter.csv', index_col=0)
    converter = {0: 'blank'}
    for _, row in df.iterrows():
        converter[row['id']] = row['name']
    return converter

# @param: season is 2021 format
# @return: starting team and gw1 pick team for the player who finished at this overall rank
def get_league_player_starting_team(season, rank, league, interval):
    player_columns = [f'player_{i}' for i in range(1,16)]
    group = (int(rank)-1) // interval
    start, end = (interval * group) + 1, interval*(group + 1)
    meta_path = DROPBOX_PATH + f"Human_Seasons/{season}/{league}_{start}-{end}/meta.csv"
    df = pd.read_csv(meta_path, index_col=0)
    players = df.loc[df['rank']==rank][player_columns].to_numpy()[0].tolist()
    chip = this_week_chip(1, df)

    gw_path = DROPBOX_PATH + f"Human_Seasons/{season}/{league}_{start}-{end}/weekly.csv"
    df = pd.read_csv(gw_path, index_col=0)
    df = df.loc[df['rank']==rank]
    captain, vcaptain = df.loc[df['gw']==1][['captain', 'vcaptain']].to_numpy()[0]
    bench = df.loc[df['gw']==1][[f'bench{x}' for x in range(1,5)]].to_numpy()[0].tolist()

    pick_team = chip, captain, vcaptain, bench 
    return players, pick_team

    
from constants import DROPBOX_PATH
import pandas as pd
# @param: for saving purposes, params is dict of bools (transfers, each chip, change_captain)
# @return: save into the csv a constructed season
def construct_fake_player_season(season, league_name, params, rank='auto'):
    # get starting team with people who get at least pts_cutoff points
    def pick_random_fifteen(pts_cutoff = 10):
        num_picks = [2,5,5,3]
        all_the_lists = [gks, defs, mids, fors] = [[],[],[],[]]
        data_df = get_data_df(CENTURY, season)
        elems = list(data_df['element'].unique())
        random.shuffle(elems)
        for elem in elems:
            if data_df.loc[data_df['element']==elem]['total_points_N1'].sum() > pts_cutoff:
                all_the_lists[-1 + int(data_df.loc[data_df['element']==elem]['position'].to_numpy()[0])].append(elem)
        
        team_starters = defs[:5]+fors[:3]+gks[:1]+mids[:2]
        team_bench = gks[1:2] + mids[2:5]
        team_captain = random.choice(team_starters)
        team_vcaptain = random.choice([x for x in team_starters if x != team_captain])

        return team_starters, team_bench, team_captain, team_vcaptain

    start, end = 1, 250
    meta_path = DROPBOX_PATH + f"Human_Seasons/{season}/{league_name}_{start}-{end}/meta.csv"
    gwks_path = DROPBOX_PATH + f"Human_Seasons/{season}/{league_name}_{start}-{end}/weekly.csv"
    try:
        meta_df = pd.read_csv(meta_path, index_col=0)
        gwks_df = pd.read_csv(gwks_path, index_col=0)
        ranks = gwks_df['rank'].unique()
        if rank == 'auto' or rank in ranks:
            rank = max(ranks) + 1
    except:
        try:
            os.makedirs(DROPBOX_PATH + f"Human_Seasons/{season}/{league_name}_{start}-{end}")
        except:
            pass
        meta_df, gwks_df = pd.DataFrame(), pd.DataFrame()
        if rank == 'auto':
            rank = 1

    df = pd.DataFrame(index = range(38), columns = ['rank', 'gw', 'inb', 'outb', 'captain', 'vcaptain', 'bench1','bench2','bench3','bench4'])
    mdf = pd.DataFrame(index = range(1), columns = ['rank', 'username','total_points',\
        'wildcard1', 'wildcard2', 'bench_boost', 'triple_captain', 'free_hit'] + [f'player_{x}' for x in range(1,16)])

    if not params['transfers']:
        team_starters, team_bench, team_captain, team_vcaptain = pick_random_fifteen()
        tc = random.randint(1,39)
        bb = tc 
        while bb == tc:
            bb = random.randint(1,39)
        

        df.loc[:,'rank'] = rank
        df.loc[:,'gw'] = list(range(1,39))
        df.loc[:,'inb'] = [set()] * df.shape[0] # 38
        df.loc[:,'outb'] = [set()] * df.shape[0]
        df.loc[:,'captain'] = team_captain
        df.loc[:,'vcaptain'] = team_vcaptain
        df.loc[:,[f'bench{i}' for i in range(1,5)]] = team_bench


        mdf.loc[:,[f'player_{i}' for i in range(1,16)]] = team_starters + team_bench
        mdf.loc[:,'rank'] = rank
        mdf.loc[:,'triple_captain'] = tc
        mdf.loc[:,'bench_boost'] = bb
    

    pd.concat([gwks_df, df], axis=1).reset_index(drop=True).to_csv(gwks_path)
    pd.concat([meta_df, mdf], axis=1).reset_index(drop=True).to_csv(meta_path)




# for debugging, hopefully can compute the score to see whether it is code malfunction or data malfunction
# so we start a 5-2-3
def evaluate_season_basemindedly(season, league_name, league_interval, rank):
    
    meta_df, gwks_df = get_meta_gwks_dfs(season, league_name, league_interval, rank)
    data_df = get_data_df(CENTURY, season)

    team = set(meta_df.loc[meta_df['rank']==rank][[f'player_{i}' for i in range(1,16)]].to_numpy()[0])
    
    bb, tc = meta_df[['bench_boost', 'triple_captain']].to_numpy()[0]

    ''' ASSUMING a 1-5-2-3 ''' 
    points_dict = {}
    for gw in range(1,39):
        df = data_df.loc[(data_df['gw']==gw)&(data_df['element'].isin(team))]
        gwk_df = gwks_df.loc[gwks_df['gw']==gw]

        captain, vcaptain = gwk_df[['captain', 'vcaptain']].to_numpy()[0]
        bench = gwk_df[[f'bench{i}' for i in range(1,5)]].to_numpy()[0].tolist()
        nonminuteless = set(df.loc[df['minutes_N1']>0]['element'].to_list()) #have to do this way bcz of players who join late
        team_minuteless = team.difference(nonminuteless)
        bench_set = set(bench)
        minuteless_starters = team_minuteless.difference(bench_set)
        diff_positions = set([int(x) for x in data_df.loc[data_df['element'].isin(minuteless_starters)]['position'].unique()])
        if len(diff_positions.difference({1})) <= 1:
            cant_max_bench = True
        else:
            cant_max_bench = False

        num_minuteless_starters = len(minuteless_starters)
        bench_minuteless_players = team_minuteless.intersection(bench_set)
        points = df.loc[(~df['element'].isin(bench))]['total_points_N1'].sum()
        fpts = points
        
        if not math.isnan(bb) and bb == gw: # bench boost adapt 
            points += df.loc[df['element'].isin(bench)]['total_points_N1'].sum()

        else:
            ''' starting keeper benched? --> pts for bench keeper '''
            starting_keeper_minutes = df.loc[(~df['element'].isin(bench))&(df['position']==1)]['minutes_N1'].sum()
            keeper_started = starting_keeper_minutes != 0.0
            backup_keeper = df.loc[(df['element'].isin(bench))&(df['position']==1)]
            bench_scores =  [int(df.loc[df['element']==x]['total_points_N1'].sum()) for x in bench]
            bench.remove(int(backup_keeper['element'].to_list()[0]))
            if not keeper_started:
                points += backup_keeper['total_points_N1'].sum() #only 1
                num_minuteless_starters -=1 

            ''' then count starter benches --> pts for non benched benchwarmers '''
            poss = [int(df.loc[df['element']==x]['position'].sum()) for x in bench]
            min_bench =  [i for i,x in enumerate(bench) if x not in bench_minuteless_players ]
            bench = [x for x in bench if x not in bench_minuteless_players ]
            if cant_max_bench:
                num_minuteless_starters = min(num_minuteless_starters, 2)
            points += df.loc[df['element'].isin(bench[:num_minuteless_starters])]['total_points_N1'].sum()
        
        
        ''' (triple) captain'''
        cap_started = True
        if df.loc[df['element']==captain]['minutes_N1'].to_numpy()[0] == 0.0:
            cap_started = False
            captain = vcaptain
        cpts = df.loc[df['element']==captain]['total_points_N1'].to_numpy()[0]
        if not math.isnan(tc) and tc == gw: # bench boost adapt 
            points += cpts
        points += cpts

        points_dict[gw] = points 
        #print(f'gw {gw}: keep? {keeper_started} , cap? {cap_started} , cap: {cpts} , {num_minuteless_starters + int(not keeper_started)} subs and bench scored {bench_scores} pos {poss} and who played is {min_bench} fpts {fpts} pt {points}')
    return points_dict


#compute and save all the transfer markets with extra gw column
# under the assumption all players are healthy
# @param: all_suites [[all_field_suites, all_keeper_suites]]
from Oracle import avg_transfer_markets, keeper_outfield_split, handle_field_players, handle_keepers
def compute_all_transfer_dfs(season, folder, data_df, all_suites):

    name_df = data_df[['element', 'name']].drop_duplicates()
    for model in all_suites[0] + all_suites[1]:
        print(f'computing model {model}')
        secret_dict = make_secret_dict(season, model)
        #print(secret_dict)
        mod_type = ('keeper' if model in all_suites[1] else 'field')
        health_df = pd.DataFrame([[x, 'a'] for x in data_df['element'].unique()], columns=['element', 'status'])
        all_gwks = []
        for gw in range(2,39):
            print(f'gw{gw}')
            current_gw_stats = data_df.loc[data_df['gw']==gw]
            gw = current_gw_stats['gw'].to_list()[0] 
            keepers, health_keepers, outfield, health_outfield = keeper_outfield_split(current_gw_stats, health_df)
            if mod_type == 'field':
                full_transfer_market = handle_field_players([f'{season}/{model}'], outfield, health_outfield, gw, name_df, preloaded=secret_dict, model_folder='Yearly')
            elif mod_type == 'keeper':
                full_transfer_market = handle_keepers([f'{season}/{model}'], keepers, health_keepers, gw, name_df, preloaded=secret_dict, model_folder='Yearly')
            full_transfer_market['gw'] = gw
            all_gwks.append(full_transfer_market)

        ftm_all_gwks = pd.concat(all_gwks).reset_index(drop=True)
        try:
            os.makedirs(folder)
        except:
            pass
        ftm_all_gwks.to_csv(folder + model+'.csv')

# season is int
def generate_x_random_starting_teams(season, x):
    if str(season) in os.listdir(DROPBOX_PATH + "Human_Seasons/"):
        NUM_PLAYERS_SCRAPED = 8500
        if x > 8500 - 1:
            raise Exception("requesting too many starting teams compared to what we saved 8499")
        ranks = []
        for _ in range(x):
            rank = None
            while(rank is None or rank in ranks):
                rank = random.randint(1, 8499)
            ranks.append(rank)
        starter_packs = []
        for rank in ranks:
            joined_later_than_gw_1 = True #som epeople can still make leaderboard joining in week 2
            while joined_later_than_gw_1:
                meta, gwk = get_meta_gwks_dfs(season, 'Overall', 250, rank)
                starting_team = meta[[f'player_{i}' for i in range(1,16)]].to_numpy()[0].tolist()
                bench = gwk.loc[gwk['gw']==1][[f'bench{i}' for i in range(1,5)]].to_numpy()[0].tolist()
                captain, vcaptain = gwk.loc[gwk['gw']==1][['captain', 'vcaptain']].to_numpy()[0]
                if any([math.isnan(x) for x in starting_team + bench + [captain, vcaptain]]):
                    rank = (rank + 1 if rank < NUM_PLAYERS_SCRAPED // 2 else rank - 1)
                else:
                    chip = this_week_chip(1, meta)
                    starting_pick_team =  chip, captain, vcaptain, bench 
                    starter_packs.append([starting_team, starting_pick_team])
                    joined_later_than_gw_1 = False

    else: #don't have their season data so make random teams
        print('making a random team')
        TARGET_PRICE = [980, 1000] #cost between 98 and 100 mil
        pos_dict = {
            1: 2, 2: 5, 3: 5, 4:3
        }
        min_pos_dict = {
            1:1, 2:3, 3:2, 4:1
        }
        max_pos_dict = {
            1:1, 2:5, 3:5, 4:3
        }

        def meets_requirements(team):
            if team['value'].sum() >= TARGET_PRICE[0] and team['value'].sum() <= TARGET_PRICE[1]:
                team_counts = get_counts(team, 'team')
                pos_counts = get_counts(team, 'position')
                for pos, req in pos_dict.items():
                    if pos_counts[pos] != req:
                        return False
                if max(list(team_counts.values())) > 3:
                    return False 
                return True
            else:
                return False

        def remove_best(team, position):
            total_cost = team['value'].sum()
            count_dict = get_counts(team, 'team')
            closeness = {}
            choice_index = None
            for i, player in team.loc[team['position']==position].iterrows():
                if count_dict[player['team']] > 3:
                    choice_index = i
                team_cost = total_cost - player['value']

                closeness[i] = max(TARGET_PRICE[0] - team_cost, team_cost - TARGET_PRICE[1])
            if choice_index is None:
                choice_index = min(closeness, key=closeness.get)
            return team.drop([choice_index], axis=0)

        ranks = [0] * x
        starter_packs = []
        data_df = get_data_df(20, season)
        full_df = data_df.loc[data_df['gw']==1][['element','position','team','value']]

        for _ in range(x):
            team = pd.DataFrame()
            df = full_df.sample(frac=1).reset_index(drop=True)
            for pos, num_players in pos_dict.items():
                pos_players = df.loc[df['position']==pos].reset_index(drop=True).loc[:num_players-1, ['element','position','team','value']]
                team = pd.concat([team, pos_players])
            
            not_suceeded, while_tries = True, 0
            while not_suceeded:
                df = full_df.sample(frac=1).reset_index(drop=True)
                for _, player in df.iterrows():
                    new_player = player[['element','position','team','value']]

                    if new_player['element'] not in team['element'].to_list():
                        team = pd.concat([team.T, new_player], axis=1).T.reset_index(drop=True)
                        team = remove_best(team, new_player['position'])

                    if meets_requirements(team):
                        not_suceeded = False
                        break

                while_tries += 1
                if while_tries > 10:
                    print(team, team['value'].sum())
                    raise Exception("Not able to construct a random team")

            '''getting pick team now that we have a valid starting team -- most expensive players'''
            team = team.sort_values(by='value').reset_index(drop=True) #ascending sort
            keeper_bench = [team.loc[team['position']==1]['element'].to_list()[0]] #just the keeper
            pos_indices = team['position'].to_list()
            elem_pos = {}
            for i, row in team.iterrows():
                pos = row['position']
                if pos in (2,3,4):
                    if pos in (2,4) and list(elem_pos.values()) == [pos, pos]:
                        continue 
                    elem_pos[row['element']] = pos 
                if len(elem_pos) == 3:
                    break

            player_bench = list(reversed(team.loc[team['element'].isin(elem_pos)]['element'].to_list()))
            bench = keeper_bench + player_bench
            vcaptain, captain = team.loc[~team['element'].isin(bench)]['element'].to_list()[-2:]

            starting_team = team['element'].to_list()
            starting_pick_team =  'none', captain, vcaptain, bench 
            starter_packs.append([starting_team, starting_pick_team])

    return starter_packs, ranks


# making this secret dict for the model_suite so that we can overload make_model func and not have to load every tim
from Oracle_helpers import load_model
def make_secret_dict(season, model_suite):
    field_positions = ['forwards','midfielders', 'defenders']
    if model_suite == 'dgw':
        model_suites = ['dgw', 'no_dgw', 'dgw_upcoming', 'no_dgw_upcoming']
    elif model_suite == 'individuals':
        model_suites = field_positions
    elif model_suite == 'sparse_individuals':
        model_suites = [f'{x}_sparse' for x in field_positions]
    else:
        model_suites = [model_suite]

    secret_dict = {}
    for model_suite in model_suites:
        season_modelsuite = f'{season}/{model_suite}'
        print(DROPBOX_PATH + f"models/Yearly/{season_modelsuite}")
        os.chdir(DROPBOX_PATH + f"models/Yearly/{season_modelsuite}")
        secret_dict[season_modelsuite] = {}
        for filename in os.listdir():
            #print(filename)
            secret_dict[season_modelsuite][filename] = load_model(filename)
        
    return secret_dict

# take a list of precomputed model types and save the weekly averages of them into name
def create_special_model_combos(precomputed_folder, name, combos):
    models = [pd.read_csv(precomputed_folder + model + '.csv', index_col=0) for model in combos]
    for i in range(len(models)):
        models[i]['indicator'] = i
    
    transfer_markets = []

    for gw in range(1, 39):
        gw_models = [df.loc[df['gw']==gw] for df in models]
        tm = avg_transfer_markets(gw_models)
        tm['gw'] = gw
        transfer_markets.append(tm)

    df = pd.concat(transfer_markets).reset_index(drop=True)
    df.to_csv(precomputed_folder + name+'.csv')



# inside of simulate season we get the required data to enter pretty print the gw
def prepare_for_pretty_print(season, gw, data_df, name_df, team, bench, transfer, captain, vcaptain):

    def make_name_col_turn_opp_and_rename_cols(gw_data, df):
        def get_teamname(row):
            team_converter = id_to_teamname_converter_saved(f'20{str(season)[:2]}-{str(season)[2:]}')
            opp = row['opponent']
            if opp == 0:
                return team_converter[opp]

            opp_names = ''
            while opp > 0:
                if len(opp_names) > 0:
                    opp_names += ','
                this_opp = ((opp-1) % 20)+1
                opp_names += team_converter[this_opp]
                opp = (opp-1) // 20
            return opp_names


        name = df.apply(lambda x: gw_data.loc[gw_data['element']==x['element']]['name'].to_list()[0], axis=1)
        name.name = 'name'
        opponent = df.apply(lambda row: get_teamname(row), axis=1)
        opponent.name = 'opponent'
        df = df.drop('element', axis=1)
        df.columns = ['opponent', 'position', 'points']
        df = pd.concat([df, name], axis=1)
        df.loc[:,'opponent'] = opponent
        return df

    gw_data = data_df.loc[data_df['gw']==gw]
    wk_transfer_names_and_points = [[(name_df.loc[name_df['element']==element]['name'].to_list()[0], gw_data.loc[gw_data['element']==element]['total_points_N1'].to_list()[0])\
        for element in direction] for direction in transfer ]
    wk_transfer_names_and_points = [[(name, int(points)) for (name, points) in direction] for direction in wk_transfer_names_and_points]
    #('name', 'opponent', 'position', 'points') 
    field_name_df = gw_data.loc[gw_data['element'].isin(set(team['element'].to_list()).difference(bench))][['element','opponent', 'position', 'total_points_N1']]
    field_name_df = make_name_col_turn_opp_and_rename_cols(gw_data, field_name_df)
    #gw_data = gw_data.set_index('element').iloc[bench,:].reset_index() # to preserve order
    gw_data = gw_data.iloc[pd.Index(gw_data['element']).get_indexer(bench)]# to preserve order
    bench_name_df = gw_data.loc[gw_data['element'].isin(bench)][['element','opponent', 'position', 'total_points_N1']]
    bench_name_df = make_name_col_turn_opp_and_rename_cols(gw_data, bench_name_df)

    captain_name =name_df.loc[name_df['element']==captain]['name'].to_list()[0]
    vcaptain_name = name_df.loc[name_df['element']==vcaptain]['name'].to_list()[0]
    return wk_transfer_names_and_points, field_name_df, bench_name_df, captain_name, vcaptain_name

# Put lists in a string container for storing in df with news of upcoming deprecation of ragged df
def stringify_if_listlike(val):
    if type(val) in (list, set, tuple):
        return str(val)
    else:
        return val


if __name__ == '__main__':
    #construct_fake_player_season(2021, 'fake_season', {'wildcard1':False, 'wildcard2':False, 'free_hit': False, 'change_captain':False, 'transfers': False, 'bench_boost':True, 'triple_captain':True})
    #pts = evaluate_season_basemindedly(2021, 'fake_season', 250, 1)
    #print(pts, f'\n Total Points: {sum(pts.values())}')

    ''' WE PRECOMPUTED THE MODELS FOR USE IN WILDCARD FUNCTION ''' # WARNING, USING MODEL THAT WAS TRAINED ON THESE YEARS
    #### WE HAVE WRITTEN OVER THIS A LITTLE TO JUST DO THE KEEPER MODELS 
    for season in [2122]:# (1718, 1617, 1819):
        print('doing season ', season)
        data_df = get_data_df(20, season)
        #all_suites = [FIELD_MODELS, KEEPER_MODELS]
        all_suites = [[],KEEPER_MODELS]
        compute_all_transfer_dfs(season, TM_FOLDER_ROOT + str(season) + '/', data_df, all_suites)
        
        #special_precomputes = {
        #    'field_all_batch': FIELD_MODELS, 'keeper_all_batch': KEEPER_MODELS, 'field_early_transfer_batch':FIELD_MODELS_EARLY
        #}
        special_precomputes = {'keeper_all_batch': KEEPER_MODELS}
        for name, combos in special_precomputes.items():
            create_special_model_combos(TM_FOLDER_ROOT + str(season) + '/', name, combos) 
    