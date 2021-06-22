# page containing necessary constants for import across different files

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

DATAHUB = "_____ PATH TO DATAHUB DATASET ____ "
DROPBOX_PATH = "____ PATH TO ROOT FOLDER FOR ALL DATA STORAGE ____"

MANUALLY_REDO_WEEKS = [] # if your data is messed up somehow 

################### ------You change above this line------- #######################
######################### ---------------------------- ############################

VASTAAV_NO_RESPONSE_WEEKS = []
VASTAAV_ROOT = r'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/'
MANUAL_VASTAAV_ROOT = DROPBOX_PATH + "manual_vaastav/"
UPDATED_TRAINING_DB = DROPBOX_PATH + "updated_training_dataset.csv"
TRAINING_WITH_TEAMS = DROPBOX_PATH + "training_dataset_withteaminfo.csv"
TRAINING_NO_TEAMS = DROPBOX_PATH + "training_dataset_noteaminfo.csv"
TRAINING_WITH_TEAMS_ONEHOT = DROPBOX_PATH + "training_withteam_onehotposition.csv"
TRANSFER_MARKET_SAVED = DROPBOX_PATH + 'full_transfer_market.csv'

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