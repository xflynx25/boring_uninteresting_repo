# page containing necessary constants for import across different files
from private_versions.private_constants import DROPBOX_PATH, RAPID_API_KEY, WEBDRIVER_PATH, \
    WEBDRIVER_BINARY_LOC

# FPL SEASON INFORMATION
CENTURY = 20
INT_SEASON_START = 2023
STRING_SEASON = f'{INT_SEASON_START}-{str(INT_SEASON_START+1)[2:]}'
WILDCARD_2_GW_STARTS = {'2023-24':18, 2324: 18, '2022-23':18, 2223: 18, '2021-22': 17, 2122: 17,'2020-21': 17, 2021: 17, '2017-18': 17, 1718: 17,\
            '2016-17': 17, 1617: 17, '2018-19': 17, 1819: 17, }#we manually write this in for fetching player data, ONLY REALLY KNOW FOR 2021 and after

# last date of the wildcard that you personally want, check to see if should be replaced with the above
MANUAL_WILDCARD_DATES = (16, 38)


MAX_FORWARD_CREATED = 6 #10 #at least
WILDCARD_DEPTH_PARAM = 4#5 

NO_CAPTCHA = False 

#VASTAAV_NOT_FUNCTIONING = False
MANUALLY_REDO_WEEKS = []
VASTAAV_NO_RESPONSE_WEEKS = []



''' Manual Execution Control '''
NUM_PLAYERS_ON_SCOREBOARD = 50

FORCE_MODERN_WILDCARD = False
DONT_TRY_TO_PATCH_ODDS = True # set to True unless you just updated the dataset by pulling from: https://www.football-data.co.uk/
ALLOW_AUTOMATED_PATCHING = False

TRANSFER_MARKET_VISUALIZATION_ROUNDING = 3
SCOREBOARD_PRINT_VERSION = 'new'
FORCE_MOVE_TODAY = False
PICK_TEAM_ONLY = False

MAX_QUEUED_USERS = 15

DEFAULT_REFERENCE_USER = 'Athena-v1.2p'

# big dict that can be referenced for print statements
VERBOSITY = {
    'misc': True,
    'odds_matches': False,
    'Accountant_Main_Loop_Function_Notifiers': True, 
    'Previous_Points_Calculation_Info': False, 
    'squad': True,
    'odds': True,
    'odds_important': True,
    'playercounter': False,
    'brain': False,
    'brain_important': True, 
}
SCOREBOARD_VERBOSITY = 0.1



"""NOTHING MOST PEOPLE WILL TOUCH BELOW HERE"""

# PATHS
RAPID_API_HOST = "api-football-v1.p.rapidapi.com"
DATAHUB = DROPBOX_PATH + r"datahub/Base_Dataset_1_3_6.csv"
VASTAAV_ROOT = r'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/'
MANUAL_VASTAAV_ROOT = DROPBOX_PATH + "manual_vaastav/"
UPDATED_TRAINING_DB = DROPBOX_PATH + "updated_training_dataset.csv"
TRAINING_WITH_TEAMS = DROPBOX_PATH + "training_dataset_withteaminfo.csv"
TRAINING_NO_TEAMS = DROPBOX_PATH + "training_dataset_noteaminfo.csv"
TRAINING_WITH_TEAMS_ONEHOT = DROPBOX_PATH + "training_withteam_onehotposition.csv"
TRANSFER_MARKET_SAVED = DROPBOX_PATH + 'full_transfer_market.csv'
POINT_HISTORY_REFERENCE_PATH = DROPBOX_PATH + '/Human_Seasons/Reference_Point_Markers/mat_the_w.csv'
TM_FOLDER_ROOT = DROPBOX_PATH + "Simulation/athena_Simulation/transfer_markets/" #folder to read in the premade transfer markets

# STATISTICS
CORE_STATS = ['assists', 'goals_scored', 'goals_conceded', 'bonus', 'bps', 'clean_sheets','influence',\
    'creativity', 'threat', 'ict_index','yellow_cards', 'red_cards', 'own_goals','penalties_missed',\
    'penalties_saved', 'saves','minutes','total_points','transfers_in','transfers_out','transfers_balance']
NEW_STATS = ['max_team_pts_concession_pos_1', 'max_team_pts_concession_pos_2', 'max_team_pts_concession_pos_3',\
     'max_team_pts_concession_pos_4', 'avg_team_pts_concession_pos_1', 'avg_team_pts_concession_pos_2',\
      'avg_team_pts_concession_pos_3', 'avg_team_pts_concession_pos_4']
BOOLEAN_STATS = ['decent_week', 'good_week', 'great_week', 'lasted_long', 'substitute', 'absent']

accountant_core = ['gw', 'element', 'team', 'position', 'value', 'selected',\
    'transfers_in','transfers_out','transfers_balance']
accountant_team = ['gw', 'team', 'opponent', 'day', 'hour', 'was_home', 'oddsW', 'oddsD', 'oddsL']

database_core = ['season','gw', 'name', 'element', 'team', 'position', 'value', 'hour','selected',\
    'transfers_in','transfers_out','transfers_balance']
database_team = ['season', 'gw', 'team', 'opponent', 'day', 'was_home', 'oddsW', 'oddsD', 'oddsL']

api_stats_team = ['FTgf', 'Sf', 'STf', 'Cf', 'Ff','FTga', 'Sa', 'STa', 'Ca', 'Fa']

###OLD_CORE_STATS = ['assists', 'goals_scored', 'goals_conceded', 'attempted_passes', 'big_chances_created', 'big_chances_missed',\
###    'bonus', 'bps',	'clean_sheets', 'dribbles',	 'clearances_blocks_interceptions', 'completed_passes', 'influence',\
###    'creativity', 'threat', 'ict_index', 'fouls', 'key_passes', 'open_play_crosses', 'offside', 'recoveries', 'big_errors',\
###    'yellow_cards', 'red_cards','tackled', 'tackles', 'target_missed', 'own_goals', 'penalties_conceded','penalties_missed',\
###    'penalties_saved', 'saves','winning_goals', 'minutes','total_points','transfers_in','transfers_out','transfers_balance']


'''for scraping postseason'''
LEAGUE_FETCHING_LOGIN_CREDENTIALS = ['athenav1.0a@gmail.com', 'Alphafpl2022!', 6140897]#EMAIL, PASSWORD, TEAM_ID
LEAGUE_FETCHING_NUM_PLAYERS_ON_PAGE = 50
WILDCARD_2_GW_START = 19


# BAD CODE STRUCTURING REQUIRES ACCESS TO THIS, BUT PEOPLE NEED TO CHANGE IT
LAST_GAMEWEEK = 38
def change_global_last_gameweek(gw):
    global LAST_GAMEWEEK
    LAST_GAMEWEEK = gw

# just for speeding evaluator
def change_wildcard_depth(depth):
    global WILDCARD_DEPTH_PARAM
    WILDCARD_DEPTH_PARAM = depth

# Something like 
# If VERBOSITY['squad']:
#       print('the name is ', his_name)
# becomes
# truthprint('squad', 'the name is ', his_name)
def truthprint(casestring, *printargs):
    if VERBOSITY[casestring]:
        print(*printargs)