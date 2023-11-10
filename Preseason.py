'''
get a particular data point by reading from the withoutteam dataset

this gives the next n gameweek points

and the regressor are data from players/player_history.csv ---- some stats from last year, some from total average
some from average of years with greater than 0, also team and opponent data. This is a little sparse...but is actually
probably enough (lacking creativity and ict index etc on the 2015-2016 and before, as well as lacking odds stuff for the 2019-2020 season)
although a prediction on just two seasons might be just good enough

probably don't have good enough data to do anything more than just regressing the two years and only doing based on the last season. 
Actually we can probably do totals for all of the seasons. 
However, we need to find some way to get a consistent fixture difficulty system which will take time
So probably need to put preseason to the side for now, come back if have time. 

It lastly doesn't even seem like it will work that welll as a learning problem unless we can figure out how to 
get odds, transfer information, team stats, opponent stats. Otherwise will just do a very bland model. More weight to obviously 
good things and vice versea.

There is a page on the premier league website that has good past season info 
https://www.premierleague.com/players/5110/Pierre-Emerick-Aubameyang/stats?co=1&se=210
'''

"""
New strat 8/4/22:
1) Several categories,  where the players are ranked
    * total_points, bps, ict_index, creativity, influence, threat
        ** by itself, and also / minutes played
    * minutes
    * Ownership currently
    * Salience in FantasyFootballScout.co.uk
    * Salience in twitter 
    * Can we rip preseason stats from somewhere?
2) Human inputs their relative values on these categories
3) Human inputs information about their bench vs starter value
4) optimizing algorithm to minimize sum of (rank*category_value)
    and then fielding the optimal team, and then taking it *is_starter_value
    (whole optimization is done together. Some more thought may be needed)
5) remember to throw out status != 'a'
"""
from constants import STRING_SEASON, VASTAAV_ROOT
from Requests import get_df_from_internet
import pandas as pd
import numpy as np 
from Brain import best_wildcard_team, pick_team


"""SCORING FUNCTIONS
Interface: return the transfer market, with score as both expected points
"""
# original:  ranked * weighting absolute (1000 - ans)
def scoring_1(col_weights, df, meta_stats_df):
    scorecard = {id: 0 for id in df['id']}

    for col, weight in col_weights.items():
        df = df.sort_values(col, axis=0, ascending=False)
        for rank, id in zip(range(1, df.shape[0]+1), df['id']):
            scorecard[id] += rank*weight
    scoredf = pd.DataFrame([[key, val] for key, val in scorecard.items()], columns=['id', 'score'])
    
    # Now convert to the transfer market format
    eval_df = meta_stats_df.merge(scoredf, how='inner',  on='id')
    eval_df = eval_df.sort_values('score', axis=0, ascending=True) #the best person will be at the top number 0
    eval_df.loc[:, 'score'] = 1000 - eval_df['score']
    eval_df.loc[:, 'expected_pts_N1'] = eval_df['score']
    eval_df.columns = ['element', 'status', 'value', 'position', 'team', 'expected_pts_N1', 'expected_pts_full']
    return eval_df
    
    
# double ranked:  ranked * weighting , ranked again  (1000 - ans)
def scoring_2(col_weights, df, meta_stats_df):
    scorecard = {id: 0 for id in df['id']}

    for col, weight in col_weights.items():
        df = df.sort_values(col, axis=0, ascending=False)
        for rank, id in zip(range(1, df.shape[0]+1), df['id']):
            scorecard[id] += rank*weight
    scoredf = pd.DataFrame([[key, val] for key, val in scorecard.items()], columns=['id', 'score'])
    
    # Now convert to the transfer market format
    eval_df = meta_stats_df.merge(scoredf, how='inner',  on='id')
    eval_df = eval_df.sort_values('score', axis=0, ascending=True) #the best person will be at the top number 0
    eval_df.loc[:, 'score'] = eval_df.reset_index().index
    eval_df.loc[:, 'score'] = 1000 - eval_df['score']
    eval_df.loc[:, 'expected_pts_N1'] = eval_df['score']
    eval_df.columns = ['element', 'status', 'value', 'position', 'team', 'expected_pts_N1', 'expected_pts_full']
    return eval_df

# An actual score: % of top * weighting
def scoring_3(col_weights, df, meta_stats_df):
    scorecard = {id: 0 for id in df['id']}

"""END SCORING FUNCTIONS """

# grab the data
filename = VASTAAV_ROOT + STRING_SEASON + '/players_raw.csv'
players_df = get_df_from_internet(filename)

'''The things we may want to consider '''
# META STUFF = ['id','now_cost','status','element_type', 'web_name', 'first_name', 'second_name']
# STATIC INFORMATION ['value_season', 'selected_by_percent', 'points_per_game', 'minutes']
# MINUTE PRE-INFORMATION (CLASSIC STATS) ['bonus', 'bps', 'creativity',  'threat', 'total_points', 'ict_index','influence', 'yellow_cards']
# POSITION SPECIFIC INFO {'clean_sheets': [0,1], 'goals_conceded':[0,1], 'saves':[0]}
# OTHERS = ['penalties_order']

# CONSTANT THINGS
meta_stats = ['id','status', 'now_cost','element_type', 'team']
name_df = players_df[['id', 'web_name']] 
name_df.columns = ['element', 'name']
health_df = players_df[['id', 'status']] 
health_df.columns = ['element', 'status']

'''-------------------------INPUT-------------------------------------------'''
# declare relavent info  ** YOU CHANGE ** 
static_stats = ['selected_by_percent', 'minutes']
classic_stats = ['bonus', 'total_points', 'ict_index'] #, 'yellow_cards']
minutes_classic_stats = [f'min_{col}' for col in classic_stats] #classic_stats = nonmetastatic_stats
new_player_stats = ['selected_by_percent']

# declare weights ** YOU CHANGE ** 
new_player_premium = 1.25 # needs to be 1.25 as good to get if didn't play minutes last year 
bench_weight = 0.4
col_weights = {'selected_by_percent':3, 'minutes': 1}
col_weights.update({
    col: 1 for col in minutes_classic_stats
})

# choose scoring method
scoring_function = scoring_1
'''-------------------------END_INPUT---------------------------------------'''
# remove status != 'a'
df = players_df.loc[players_df['status']=='a']

# deal with new players (132 healthy new players ... big portion)
df = df.loc[df['minutes'] > 0]
new_multiplier = (1 / new_player_premium) * sum(col_weights.values()) / \
    sum({key:val if key in new_player_stats else 0 for key,val in col_weights.items()}.values())


# make the df with the judging columns 
meta_stats_df = df[meta_stats]
static_stats_df = df[static_stats]
classic_df = df[classic_stats]
minutes_classic_df =  df[classic_stats].div(df['minutes'], axis=0).dropna()
minutes_classic_df.columns = minutes_classic_stats # the people who didn't play last year will just have multipliers based on their salience
df = pd.concat([meta_stats_df, static_stats_df, classic_df, minutes_classic_df], axis=1)

# Calling Score Func
full_transfer_market = scoring_function(col_weights, df, meta_stats_df)

# Getting Players
#best_wildcard_team(full_transfer_market, sell_value, bench_factor, free_hit=False,\
#allowed_healths=['a'], visualize_names=False, name_df = None)
# full_transfer_market = status, element, position, team, value, expected_pts_N1, expected_pts_full
final_players, wildcard_pts = best_wildcard_team(full_transfer_market, 1000, bench_weight, visualize_names=False, name_df = name_df)

print('Using Scoring Method {score_method_number}')
print(f'With bench value of {bench_weight} and \n col weights of {col_weights} \n \n Team IS: \n')
"""
for elem in final_players['element']:
    print(name_df.loc[name_df['element'] == elem]['name'].to_numpy()[0])
"""

# PRETTY DISPLAY
def get_name(name_df, element):
    try:
        return name_df.loc[name_df['element']==element]['name'].to_numpy()[0]
    except:
        return 'Invalid ID'

(starters, bench_order, captain, vice_captain), cap_pts, bench_score = pick_team(final_players, health_df, with_keeper_bench=True)
print('\nStarters\n---------')
for starter in starters:
    print(get_name(name_df, starter))
print('\nBench\n---------')
for bench_player in bench_order:
    print(get_name(name_df, bench_player))
print('\nCaptain/VC\n-----------')
print(f'{get_name(name_df, captain)} / {get_name(name_df, vice_captain)}')

'''next steps'''
# take care of nans
# then run the search
# add the positional things 
# if time do ffscout but prob no time