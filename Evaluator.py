"""
STATE MACHINE
f (season, starting_team, g(processed_data, current_team, sell_value, free_transfers, chip_availabilities)) 
    --> season score 
!-! We keep track of team as what we bought them for, recalculate their sell price before send to transfer func
"""
from constants import DROPBOX_PATH
import pandas as pd
import time

# @param: 
##  - starting_team is list of elements 
##  - transfer func ^^^^
##  - when transfer is early, late, or random (where can be any between the two weeks, may help us because this is uncertain knowledge for comp)
# squad is list of (element, sell_value)
# team is df of element, buy price, current_price, sell_price
# we keep track of buy prices, and then calculate other 
def simulate_season(data_df, starting_team, transfer_function, when_transfer='late', starting_sv = 100.0):
    data_gw1 = data_df.loc[data_df['gw']==1]
    team = pd.DataFrame(index=range(len(starting_team)),columns=['element','position', 'purchase','now','sell'])
    team.loc[:, 'element'] = starting_team
    team.loc[:, 'purchase'] = team['element'].apply(lambda x: data_gw1.loc[data_gw1['element']==x]['value'].to_numpy()[0])
    sv = starting_sv
    itb = sv - team['purchase'].sum()
    ft = 1
    chip_status = {'wildcard': (True, 19), 'freehit': True, 'bench_boost': True, 'triple_captain': True}

    scores = {}
    for gw in range(1,39):
        ## Adjust Prices ##
        this_gw_data = data_df.loc[data_df['gw']==gw]
        if when_transfer == 'early' and gw != 1:
            gwk_data = data_df.loc[data_df['gw']==gw-1]
        elif when_transfer == 'late' or gw == 1:
            gwk_data = this_gw_data
        team.loc['now'] = team['element'].apply(lambda x: gwk_data.loc[gwk_data['element']==x]['value'].to_numpy()[0])
        team.loc['sell'] = (team['now'] - team['purchase'])
        team.loc['sell'] = (2 * team['purchase'] + team['sell'].apply(lambda x: [x if x >= 0 else 2*x][0]) ) // 2
        sv = team['sell'].sum() + itb

        ## TRANSFER ##
        if gw > 1:
            squad = team[['element', 'sell']].to_list()
            transfer, chip, captain, vcaptain, bench = transfer_function(this_gw_data, gw, squad, sv, ft, chip_status)
            inbound, outbound = transfer 

            ## EVOLVE TEAM ##
            if len(inbound):
                new_players = pd.DataFrame(index=range(len(inbound)),columns=['element','position', 'purchase','now','sell'])
            old_players = team.loc[~team['element'].isin(outbound)]
            new_players.loc[['element', 'position', 'purchase']] = gwk_data.loc[gwk_data['element'].isin(inbound)][['element', 'value']]
            team = pd.concat([old_players, new_players],axis=0)
            if chip == 'wildcard':
                chip_status['wildcard'][0] = False
            elif chip:
                chip_status[chip] = False
            if gw == 19:
                chip_status['wildcard'] = (True, 38)
            itb = sv - team['sell'].sum()
            ft = [1 if chip in ('wildcard', 'freehit') else min(max(0, ft - len(inbound) + 1), 2)][0]


        # evaluate the team 
        results = this_gw_data[['element', 'position', 'minutes_N1', 'total_points_N1']]
        scores[gw] = score_round(results, set(team['element'].to_list()), chip, captain, vcaptain, bench)

    return scores

    
#@ return some_dict 
#@ param: bench is a list
def score_round(results, team, chip, captain, vcaptain, bench_list):
    bench = set(bench_list)
    score = 0
    minuteless = results.loc[(results['element'].isin(team['element']))&(results['minutes_N1']==0.0)]
    field_minuteless = set(minuteless.loc[(~minuteless['element'].isin(bench))]['element'].to_list())
    bench_minuteless = set(minuteless.loc[(minuteless['element'].isin(bench))]['element'].to_list())
    num_subs = 0
    subs_available = len(minuteless)

    ## CAPTAIN
    if captain in field_minuteless:
        captain = vcaptain
    score += results.loc[results['element']==captain]['total_points_N1'].to_list()[0]
    if chip =='triple_captain':
        score *= 2

    score += sum(results.loc[results['element'].isin(team['element'])]['total_points_N1'])
    starters = set(team['element'].to_list()).difference(bench)
    
    if chip != 'bench_boost':

        fields = team.loc[~team['element'].isin(bench)]['position'].to_list()
        outs = team.loc[team['element'].isin(field_minuteless)]['position'].to_list()
        bench = bench.difference(bench_minuteless)
        bench_list = [x for x in bench_list if x in bench]
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
                if pos == sub_pos or fields[pos] > requirement(pos):
                    bench_list.remove(sub)
                    found = pos
                    break
            if found:
                outs.remove(found)
                fields_dict[pos] -= 1
                fields_dict[sub_pos] = fields_dict.get(sub_pos, 0) + 1

        score -= sum(results.loc[results['element'].isin(bench_list)]['total_points_N1'])
    return score 


#@return: transfer, chip, captain, vcaptain, bench
def some_transfer_function(data, gw, squad, sv, ft, chip_status):
    pass


#@return: transfer, chip, captain, vcaptain, bench
def top_player_transfer_function(data, gw, squad, sv, ft, chip_status, rank):
    pass

    gwks_path = DROPBOX_PATH + "Human_Seasons\2021\Overall_1-250\weekly.csv"

def get_top_player_starting_team(rank):
    player_columns = [f'player_{i}' for i in range(1,16)]
    group = int(rank) // 250
    start, end = (250 * group) + 1, 250*(group + 1)
    meta_path = DROPBOX_PATH + f"Human_Seasons/2021/Overall_{start}-{end}/meta.csv"
    df = pd.read_csv(meta_path, index_col=0)
    return df.loc[df['rank']==rank][player_columns].to_numpy()[0].tolist()



if __name__ == '__main__':
    rank = 1
    coone_tf = lambda x: top_player_transfer_function(*x, rank)
    coone_starting_team = get_top_player_starting_team(rank)
    print(coone_starting_team)
    training_path = DROPBOX_PATH + r"2020-21\Processed_Dataset_2021.csv"
    df = pd.read_csv(training_path, index_col=0)
    this_season = 2021
    data_df = df.loc[df['season']==this_season]
    print(data_df.shape)
    start = time.time()
    coone_scores = simulate_season(data_df, coone_starting_team, coone_tf, starting_sv=150.0)
    print("Length = ", time.time() - start)
    print(coone_scores)