# go through top 100k or something and record their transfers every week, can use this for training
# an imitation bot.

# when you click on the player's popup, then you can see their team and position, and other stuff to 
# resolve conflicts 

# give some time between requests so don't get fpl busy to making captcha

from Requests import proper_request
from constants import DROPBOX_PATH, LEAGUE_FETCHING_LOGIN_CREDENTIALS, LEAGUE_FETCHING_NUM_PLAYERS_ON_PAGE
from selenium import webdriver
driver = webdriver.Chrome() #webdriver.Firefox()
HOSTNAME = 'https://fantasy.premierleague.com'
destination_folderpath = DROPBOX_PATH + "Human_Seasons/2021/"
login_url = HOSTNAME
EMAIL, PASSWORD, TEAM_ID = LEAGUE_FETCHING_LOGIN_CREDENTIALS

NUM_PLAYERS_ON_PAGE = LEAGUE_FETCHING_NUM_PLAYERS_ON_PAGE
LAST_GW = 38

PLAYER_CONVERTER_ONLINE_TO_ELEMENT = {}
TEAM_CONVERTER_ONLINE_TO_NAMEDF = {}
BIG_DATA_DF = None
BIG_DATA_DF_PATH = DROPBOX_PATH + "player_raw.csv"

POSITION_WORD_TO_NUMBER_DICT = {
    'GKP':1, 'DEF':2, 'MID':3, 'FWD':4
}

""" \/\/\/\/\/ #### --- END CONFIGURATION SECTION --- #### \/\/\/\/\/"""
"""------------------------------------------------------------------"""


"""     ####    MAIN CODE    ####    """
import shutil
import pandas as pd
import os
import re
import time
import random
from bs4 import BeautifulSoup
from jellyfish import levenshtein_distance


def safe_url(url):
    if 'http' not in url:
        url = HOSTNAME + url
    return url

def get_soup(url, wait_time = 1):
    url = safe_url(url)
    driver.get(url)
    time.sleep(wait_time)
    return BeautifulSoup(driver.page_source, 'html.parser')

def get_soup_and_oneclick_soup(url, click_text, wait_time = 1):
    url = safe_url(url)
    driver.get(url)
    time.sleep(wait_time)
    a = BeautifulSoup(driver.page_source, 'html.parser')
    our_link = driver.find_element_by_link_text(click_text)
    our_link.click()
    b = BeautifulSoup(driver.page_source, 'html.parser')
    return a,b 

def make_folder_safe(name):
    if not os.path.exists(name):
        os.makedirs(name)


USERNAME_ID = "loginUsername"
PASSWORD_ID = "loginPassword"
SUBMIT_BUTTON_CLASSES = ["ArrowButton-thcy3w-0","hHgZrv"]
def login_to_website():
    driver.get(login_url)
    time.sleep(.25)
    driver.find_element_by_id(USERNAME_ID).send_keys(EMAIL)
    driver.find_element_by_id(PASSWORD_ID).send_keys(PASSWORD)
    driver.find_element_by_xpath("//button[@type='submit']").submit()
    time.sleep(2)

def logout_from_website():
    driver.get(login_url)
    time.sleep(.25)
    try:
        driver.find_element_by_link_text('Sign Out').click()
        time.sleep(1)
    except:
        driver.find_element_by_class_name("Dropdown__MoreButton-qc9lfl-0").click()
        time.sleep(2)
        driver.find_element_by_link_text('Sign Out').click()
        time.sleep(1)


def has_classes(thing, classes):
    return thing.has_attr('class') and all(map(lambda x: x in thing['class'], classes))

def make_name_team_pos_df():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = proper_request("GET", url, headers=None)
    players_raw = pd.DataFrame(response.json()['elements'])
    team_id_to_name_converter = {t['code']: t['name'] for t in response.json()['teams']}
    print(players_raw.columns)
    name_df = players_raw[['id','web_name','team_code', 'element_type']]
    name_df.loc[:,'team_code'] = name_df.loc[:,'team_code'].apply(lambda x: team_id_to_name_converter[x])
    name_df.columns = ['element', 'name', 'team', 'position']
    return name_df

def str_distance(name, target):
    name, target = name.lower(), target.lower()
    min_score = float('inf')
    size = len(target) 
    buffer = ' ' * (size-1) 
    buffered_name = buffer + name + buffer 
    for start in range(len(buffered_name)-size+1):
        end = start+size
        score = levenshtein_distance(buffered_name[start:end],target)
        min_score = min(min_score, score)
    return min_score

# @param: df = has name and element columns
def find_match(df, target, tiebreak_info=None):
    scores = {}
    for _, row in df.iterrows():
        name = row['name']
        element = row['element']
        scores[element] = str_distance(name, target) 
    sorted_scores = sorted(scores, key=lambda x: scores[x])
    if len(sorted_scores) > 1 and scores[sorted_scores[0]] == scores[sorted_scores[1]]: #tie
        if tiebreak_info:
            pass #EDIT#
        else:
            print('The target is: ', target)
            for i in range(4):
                print('the top options are: ', df.loc[df['element']==sorted_scores[i]]['name'], ' with a score of ', scores[sorted_scores[i]])
            return 'tie'
    else:
        closest = sorted_scores[0]
        return closest
        

# gets the desired player, may fail if multiple players on team with same name
# if that happens return tie --> and we will provide with more info
# this takes a list of players so some could be no tie and some could tie
# @param: bad_tuples -> (name, team, pos, influence, bps)
def get_elements_from_namestrings_and_team(gw, bad_tuples, name_team_pos_df, visualize=False, tiebreak_info = None):

    bad_elements = []
    for bad_guy, his_team, pos, influence, bps in bad_tuples:
        # get target team from his_team
        if his_team in TEAM_CONVERTER_ONLINE_TO_NAMEDF:
            target_team = TEAM_CONVERTER_ONLINE_TO_NAMEDF[his_team]
        else:
            teams = name_team_pos_df['team'].unique().tolist()
            df = pd.DataFrame([teams, teams]).T
            df.columns = ['name', 'element']
            target_team = find_match(df, his_team)
            if target_team == 'tie':
                raise Exception("Couldn't clarify the team")
            if visualize:
                print('TEAM_MATCH is ', target_team)
            TEAM_CONVERTER_ONLINE_TO_NAMEDF[his_team] = target_team

        if (bad_guy, his_team, pos) in PLAYER_CONVERTER_ONLINE_TO_ELEMENT:
            match = PLAYER_CONVERTER_ONLINE_TO_ELEMENT[(bad_guy, his_team, pos)]
        else:
            this_team_df = name_team_pos_df.loc[(name_team_pos_df['team']==target_team)&(name_team_pos_df['position']==pos)]
            match = find_match(this_team_df, bad_guy, tiebreak_info=tiebreak_info)
            if match == 'tie': #resolve tie
                if not BIG_DATA_DF:
                    BIG_DATA_DF = pd.read_csv(BIG_DATA_DF_PATH, index_col=0)
                df = BIG_DATA_DF.loc[BIG_DATA_DF['gw']==gw]
                eps = .09
                possibilities = df.loc[(df['bps']==bps)&(df['influence']>=influence-eps)&(df['influence']<=influence+eps)]
                options_df = this_team_df.loc[this_team_df['element'].isin(possibilities['element'])]
                match = find_match(options_df, bad_guy, tiebreak_info=tiebreak_info)
                # debug printing
                bad_name = this_team_df.loc[this_team_df['element']==match]['name'].to_list()[0]
                print('We decided to select __', bad_name, '__ to match with the player __', bad_guy, '__')
            else:
                PLAYER_CONVERTER_ONLINE_TO_ELEMENT[(bad_guy, his_team, pos)] = match

        if match != 'tie':           
            if visualize:
                bad_name = this_team_df.loc[this_team_df['element']==match]['name'].to_list()[0]
                print('MATCH ', match, ' name is ', bad_name)
        else:
            raise Exception("There is a tie even after clarification?? Guy is ", bad_guy)
        bad_elements.append(match)
            
            

    return bad_elements


""" ### DRIVER FUNCTIONS ### """
# for a season username -> userplace & starting 15, when chips
# userplace, gw, inb (set), outb (set), chip (0-4), c, vc, bench (set)
def get_top_players(name_team_df, num_players, save_interval = 500, visualize=False, league_name = 'Overall', start_at_rank=1):
    login_to_website()
    soup = get_soup(HOSTNAME + '/leagues')
    next_url = ''
    for link in soup.find_all('a'):
        if link.get_text() == league_name:
            next_url = link['href']
            break
    
    top_players = []
    round = 1
    while len(top_players) < num_players:
        prev_url = next_url
        soup = get_soup(safe_url(next_url))
        for link in soup.find_all('a'):
            if len(top_players) == round * NUM_PLAYERS_ON_PAGE or len(top_players) == num_players:
                pass
            elif has_classes(link, ["Link-a4a9pd-1"]):# changed between days? -->,"kBzihF"]):
                top_players.append(link['href'].split('/')[2])
            
            if link.has_attr('variant') and link['variant'] == "secondary" and 'report' not in link.get_text().lower():
                next_url = link['href']
        round += 1
        if prev_url == next_url: #stop if on last page
            break

    num_players = len(top_players)

    for outer_index in range(num_players // save_interval + (num_players % save_interval > 0)):
        mdf = pd.DataFrame(index = range(min(save_interval, num_players-outer_index*save_interval)), columns = ['rank', 'username','total_points',\
            'wildcard1', 'wildcard2', 'bench_boost', 'triple_captain', 'free_hit'] + [f'player_{x}' for x in range(1,16)])
        sdfs = []
        this_chip, this_gw = '',''
        for rel_rank, player_id in enumerate(top_players[outer_index*save_interval:min((outer_index+1)*save_interval, num_players)]):
            rank = outer_index*save_interval + rel_rank + 1
            print('Rank ', rank)
            if rank < start_at_rank:
                break
            df = pd.DataFrame(index = range(LAST_GW), columns = ['rank', 'gw', 'inb', 'outb', 'captain', 'vcaptain', 'bench1','bench2','bench3','bench4'])
            player_base = HOSTNAME + '/entry/' + str(player_id) + '/'
            
            ### GET META
            usnm, tt_pts, found = None, None, False
            soup = get_soup(player_base + 'history')
            for place in soup.find_all('h2'):
                if has_classes(place, ["Entry__EntryName-sc-1kf863-0"]):#daily --> ,"ldMMkD"]):
                    usnm = place.get_text()
            for item in soup.find_all('li'):
                if has_classes(item, ["Entry__DataListItem-sc-1kf863-1"]):#daily --> ,"hMFluK"]):
                    for place in item.find_all('h5'):
                        if 'overall' in place.get_text().lower():
                            for div in item.find_all('div'):
                                tt_pts = int(div.get_text())
                                found = True
                                break
                        if found:
                            break
                if found:
                    break
            df.loc[:,'rank'] = rank
            mdf.loc[rel_rank, ['rank', 'username', 'total_points']] = rank, usnm, tt_pts

            ### WEEKLY INFO
            players_rn = []
            for gw in range(1,LAST_GW + 1):
                not_succeed, special_mistakes = True, 0
                while not_succeed and special_mistakes < 5:
                    try:
                        df.loc[gw-1, 'gw'] = gw

                        ### GET STARTING TEAM AND GET TRANSFER STUFF
                        #soup = get_soup(player_base + f'event/{gw}')
                        #list_soup = get_oneclick_soup(player_base + f'event/{gw}', "List View", wait_time = 1)
                        soup, list_soup = get_soup_and_oneclick_soup(player_base + f'event/{gw}', "List View", wait_time = 1)

                        players_prev = players_rn.copy()
                        teams, intermediate_names, positions, influences, bpss = [],[],[],[],[]
                        for player in list_soup.find_all('tr'):
                            if has_classes(player, ["ElementTable__ElementRow-sc-1v08od9-3"]):#daily --> ,"kGMjuJ"]):
                                player_name, player_team, player_pos, wk_influence, wk_bps = None, None, None, None, None
                                for div in player.find_all('div'):
                                    if has_classes(div, ["Media__Body-sc-94ghy9-2"]):#daily --> ,"eflLUc"]):#["ElementInTable__Name-y9xi40-1","dwvEEF"]):
                                        for i, div2 in enumerate(list(div.find_all('div'))):
                                            if i == 0:
                                                player_name = div2.get_text()
                                            if i == 1:
                                                word_position = list(div2.find_all('span'))[1].get_text()
                                                player_pos = POSITION_WORD_TO_NUMBER_DICT[word_position]
                                for i, stat in enumerate(reversed(list(player.find_all('td')))):
                                    if i == 3:
                                        wk_influence = float(stat.get_text())
                                    elif i == 4:
                                        wk_bps = int(stat.get_text())
                                        break
                                for img in player.find_all('img'):
                                    player_team = img['alt']
                                if all([x is not None for x in (player_name, player_team, player_pos, wk_influence, wk_bps)]):
                                    teams.append(player_team)
                                    intermediate_names.append(player_name)
                                    positions.append(player_pos)
                                    influences.append(wk_influence)
                                    bpss.append(wk_bps)
                                
                        players_rn = get_elements_from_namestrings_and_team(gw, [x for x in zip(intermediate_names, teams, positions, influences, bpss)], name_team_df, visualize=visualize)
                        if len(players_rn) != 15:
                            raise Exception("# of players != 15")
                        if gw == 1:
                            mdf.loc[rel_rank, [f'player_{x}' for x in range(1,16)]] = players_rn
                        else:
                            inb = set(players_rn) - set(players_prev)
                            outb = set(players_prev) - set(players_rn)
                            df.loc[gw-1, ['inb', 'outb']] = (inb, outb)
                        ###########################
                        ##### END LISTY SOUP ######
                        ###########################
                        ### GET PICK TEAM STUFF
                        specials = [0]*6
                        for player_class in soup.find_all('div'):
                            # CAPTAINCY
                            if has_classes(player_class,["Pitch__PitchElementWrap-sc-1mctasb-4"]):#Pitch__StyledPitchElement-sc-1mctasb-5"]):#position --> ,"igRGnf"]):
                                index = -1
                                for img in player_class.find_all('svg'):
                                    if has_classes(img,["TeamPitchElement__StyledViceCaptain-sc-202u14-2"]):#daily --> ,"jKCXOU"]):
                                        index = 5
                                    elif has_classes(img,["TeamPitchElement__StyledCaptain-sc-202u14-1"]):#daily --> ,"jUhfsE"]):
                                        index = 4
                                if index != -1:
                                    for div in player_class.find_all('div'):
                                        if has_classes(div,["PitchElementData__ElementName-sc-1u4y6pr-0"]):#--> likely health --> ,"eMnDEV"]):
                                            specials[index] = div.get_text()
                        # BENCH
                        index = 0
                        for bench in soup.find_all('div'):
                            if has_classes(bench,["Bench-sc-1sz52o9-0"]):#another daily --> ,"hWFwCm"]):
                                for div in bench.find_all('div'):
                                    if has_classes(div,["PitchElementData__ElementName-sc-1u4y6pr-0"]):#player health --> ,"eMnDEV"]):
                                        specials[index] = div.get_text()
                                        index += 1
                                    if index > 4:
                                        raise Exception("added too many to bench")
                                        
                        # correct for auto subbing
                        for autosub_table in soup.find_all('table'):
                            if has_classes(autosub_table,["Table-ziussd-1"]):#daily --> ,"fVnGhl"]):
                                for th in autosub_table.find_all('th'):
                                    if 'player in' in th.get_text().lower() or 'player out' in th.get_text().lower():
                                        for tr in autosub_table.find_all('tr'):
                                            this_sub = []
                                            for player in tr.find_all('td'):
                                                this_sub.append(player.get_text())
                                            if len(this_sub) > 0:
                                                specials[specials[:4].index(this_sub[0])] = this_sub[1]
                                        break #just care that its good
                        # change to ids
                        specials = list(map(lambda x: players_rn[intermediate_names.index(x)], specials))
                        #print('intermed names: ', intermediate_names, '\n specials: ', specials, '\n players rn: ', players_rn)

                        df.loc[gw-1, ['captain', 'vcaptain']] = specials[4:]
                        df.loc[gw-1, ['bench1','bench2','bench3','bench4']] = specials[:4]
                        sdfs.append(df)
                        #print(mdf.loc[gw-1, :])
                        not_succeed = False
                    except:
                        special_mistakes += 1


            ### GET CHIP STUFF
            not_succeed, special_mistakes = True, 0
            while not_succeed and special_mistakes < 5:
                try:
                    soup = get_soup(player_base + 'history')
                    for table in soup.find_all('table'):
                        if has_classes(table,["Table-ziussd-1"]):
                            to_find = ['date', 'name', 'active']
                            for th in table.find_all('th'):
                                this_text = th.get_text().lower()
                                for it in to_find:
                                    if it in this_text:
                                        to_find.remove(it)
                                        break
                            if len(to_find) == 0:
                                for chip_sec in table.find_all('tr'):
                                    for line in chip_sec.find_all('td'):
                                        if 'Free' in line.get_text():
                                            this_chip = 'free_hit'
                                        elif 'Wild' in line.get_text():
                                            this_chip = 'wildcard1'
                                        elif 'Bench' in line.get_text():
                                            this_chip = 'bench_boost'
                                        elif 'Triple' in line.get_text():
                                            this_chip = 'triple_captain'
                                    for link in chip_sec.find_all('a'):
                                        this_gw = int(link.get_text()[2:])

                                    if this_chip == 'wildcard1' and this_gw > 19:
                                        this_chip = 'wildcard2'
                                    mdf.loc[rel_rank, this_chip] = this_gw
                
                    not_succeed = False
                except:
                    special_mistakes += 1

        if len(sdfs) > 0:
            print('Saving !!!')
            this_folder = destination_folderpath + f'{league_name}_{outer_index*save_interval+1}-{min(num_players, (outer_index+1)*save_interval)}/'
            make_folder_safe(this_folder)
            pd.concat(sdfs, axis=0).reset_index(drop=True).to_csv(this_folder + 'weekly.csv')
            mdf.to_csv(this_folder + 'meta.csv')
    logout_from_website()

def get_all_private_league_names():
    leagues = []
    login_to_website()
    soup = get_soup(HOSTNAME + '/leagues')
    for div in soup.find_all('div'):
        if has_classes(div, ["Panel__StyledPanel-sc-1nmpshp-0"]):
            for h4 in div.find_all('h4'):
                if 'private classic leagues' in h4.get_text().lower():
                    league = div.find_all('a')[0].get_text()
                    leagues.append(league)
    logout_from_website()
    return leagues

if __name__ == '__main__':
    import traceback
    try:
        name_df = make_name_team_pos_df()
        #get_top_players(name_df, 2, save_interval = 10, visualize=False)
        get_top_players(name_df, 10000, save_interval = 250, visualize=False, start_at_rank=251)
        for pri_league in get_all_private_league_names():
            get_top_players(name_df, 5000, save_interval = 250, visualize=False, league_name=pri_league)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
    finally:
        driver.close()

    # careful about if they are doing full name, prob only want to mach the last, although
    # some guys might go by only first name. We just have to hope people don't have last name
    # as someone else's first name which is longer