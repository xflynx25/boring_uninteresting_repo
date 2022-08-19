# page containing necessary constants for import across different files
from pickle import FALSE
from malleable_constants import COMPUTER_USERNAME, C_ENTRY

CENTURY = 20
INT_SEASON_START = 2022
STRING_SEASON = f'{INT_SEASON_START}-{str(INT_SEASON_START+1)[2:]}'

NOTIFICATION_RECEIVER_EMAIL = '___ EMAIL YOU CHECK REGULARLY ____'
# If these two fields below are left blank the email will attempt to be sent with the 
# bot email, but that will not work unless you turn on 'Allow less secure apps'
NOTIFICATION_SENDER_GMAIL = '' # some email where you have turned on 'Allow less secure apps'
NOTIFICATION_SENDER_PASSWORD = ''

''' General sign-in for automated fpl website scraping '''
GENERIC_FPL_LOGIN_CREDENTIALS = ('_________', '__________', ''' INTEGER '''), #EMAIL, PASSWORD, TEAM ID
''' If also want to get private league or self information '''
LEAGUE_FETCHING_LOGIN_CREDENTIALS = GENERIC_FPL_LOGIN_CREDENTIALS # Your creds here !!
LEAGUE_FETCHING_NUM_PLAYERS_ON_PAGE = 50 # Number of players fpl leagues max out having on one page

RAPID_API_HOST = "api-football-v1.p.rapidapi.com"
RAPID_API_KEY = "_____ YOUR API KEY _______"
# other keys

# next 2 lines should be activated if running on server not home computer
#SERVER_HOME = '/home/server/'
#DROPBOX_PATH = SERVER_HOME + "fpl/data/"
DROPBOX_PATH = "____ PATH TO ROOT FOLDER FOR ALL DATA STORAGE ____"
DATAHUB = "_____ PATH TO DATAHUB DATASET ____ "

#VASTAAV_NOT_FUNCTIONING = False
MANUALLY_REDO_WEEKS = [] # if your data is messed up somehow 
VASTAAV_NO_RESPONSE_WEEKS = []

WILDCARD_2_GW_STARTS = {'2021-22': 17, 2022: 17,'2020-21': 17, 2021: 17, '2017-18': 17, 1718: 17, '2016-17': 17, 1617: 17,\
     '2018-19': 17, 1819: 17, }#we manually write this in for fetching player data, ONLY REALLY KNOW FOR 2021 and after
MAX_FORWARD_CREATED = 6 #10 #at least
WILDCARD_DEPTH_PARAM = 4 #5 

NO_CAPTCHA = False 
BACKUP_COMPUTER_WHO_RUNS_MAIN_SCRIPT = '___os username___' # for running fpl script on secondary computer
BACKUP_COMPUTER_OS_RUNS_MAIN_SCRIPT = '' # "C:/" or '/mnt/c/'


################### ------You can change above this line------- #######################
######################### ---------------------------- ############################
VASTAAV_ROOT = r'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/'
MANUAL_VASTAAV_ROOT = DROPBOX_PATH + "manual_vaastav/"
UPDATED_TRAINING_DB = DROPBOX_PATH + "updated_training_dataset.csv"
TRAINING_WITH_TEAMS = DROPBOX_PATH + "training_dataset_withteaminfo.csv"
TRAINING_NO_TEAMS = DROPBOX_PATH + "training_dataset_noteaminfo.csv"
TRAINING_WITH_TEAMS_ONEHOT = DROPBOX_PATH + "training_withteam_onehotposition.csv"
TRANSFER_MARKET_SAVED = DROPBOX_PATH + 'full_transfer_market.csv'
POINT_HISTORY_REFERENCE_PATH = DROPBOX_PATH + '/Human_Seasons/Reference_Point_Markers/mat_the_w.csv'
TM_FOLDER_ROOT = DROPBOX_PATH + "Simulation/athena_Simulation/transfer_markets/" #folder to read in the premade transfer markets

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

LAST_GAMEWEEK = 38

def change_global_last_gameweek(gw):
    global LAST_GAMEWEEK
    LAST_GAMEWEEK = gw

###OLD_CORE_STATS = ['assists', 'goals_scored', 'goals_conceded', 'attempted_passes', 'big_chances_created', 'big_chances_missed',\
###    'bonus', 'bps',	'clean_sheets', 'dribbles',	 'clearances_blocks_interceptions', 'completed_passes', 'influence',\
###    'creativity', 'threat', 'ict_index', 'fouls', 'key_passes', 'open_play_crosses', 'offside', 'recoveries', 'big_errors',\
###    'yellow_cards', 'red_cards','tackled', 'tackles', 'target_missed', 'own_goals', 'penalties_conceded','penalties_missed',\
###    'penalties_saved', 'saves','winning_goals', 'minutes','total_points','transfers_in','transfers_out','transfers_balance']

def change_wildcard_depth(depth):
    global WILDCARD_DEPTH_PARAM
    WILDCARD_DEPTH_PARAM = depth