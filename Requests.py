import requests
import re  
import io
import time
import pandas as pd
from math import log2
from constants import RAPID_API_HOST, RAPID_API_KEY, INT_SEASON_START

ENCODING1 = 'UTF-8'
ENCODING2 = 'ISO-8859-1'
HEADERS = {
        'x-rapidapi-host': RAPID_API_HOST,
        'x-rapidapi-key': RAPID_API_KEY
        }

class Custom404Exception(Exception):
    """Exception raised when you request url that isn't there"""
    pass
 
# making api calls, exponential backoff for 429
def proper_request(req_type, url, headers=None, params=None, max_requests=10, max_429_wait_time = 15*60, checking_for_404=False):
    def make_requests():
        count = 0
        response = ''
        while response == '' and count < max_requests:
            try:
                response = requests.request(req_type, url, params=params, headers=headers)
            except:
                time.sleep(5)
                count += 1
        if count == max_requests:
            raise Exception("Could not get any response at all!")
        return response

    response = make_requests()

    errors = 0
    max_requests = log2(max_429_wait_time / 15) // 1
    while response.status_code == 429: #recover from 429 errors
        print('429 response: ', response)
        time.sleep(15*2**errors)
        response = make_requests()
        errors += 1
        if errors > max_requests:
            raise Exception("Too many 429 can't break down barrier")

    if response.status_code != 200:
        if response.status_code == 404:
            raise Custom404Exception("got a 404")
        raise Exception("Response was code " + str(response.status_code))
    return response


# used for drawing csv from github
def get_df_from_internet(url):
    r = proper_request("GET", url).content    
    try:
        csv = io.StringIO(r.decode(ENCODING1))
    except UnicodeDecodeError:
        print('UTF-8 not working')
        csv = io.StringIO(r.decode(ENCODING2)) 
    return pd.read_csv(csv)

###
##
#
""" BELOW HERE ARE SPECIFIC REQUESTS FOR RAPIDAPI"""
#
##
###

# v3
# param: start season year integer yyyy, returns: league id
def get_premier_league_id(year):

    url = "https://api-football-v1.p.rapidapi.com/v3/leagues"
    querystring = {"country":"England","season":year}
    response = proper_request("GET", url, headers=HEADERS, max_requests=10, params=querystring)
    leagues = [val['league'] for val in response.json()['response']]

    for league in leagues:
        if league['name'] == 'Premier League':
            return league['id']
    raise Exception("couldn't find league id")

# league_id : 2790 for prem2020
# date_string : yyyy-mm-dd
# return: list of (id, h, a, home_g, away_g, status) for all fixtures in the league on the date 
# can return matches originally scheduled for that date, cancelled, and not yet rescheduled
def get_fixture_ids(league_id, date_string):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {"date":date_string,"league":league_id, "season": INT_SEASON_START}

    response = proper_request("GET", url, headers=HEADERS, params = querystring, max_requests=10)
    fixtures = response.json()['response']
    all_fix = []
    for fix in fixtures:
        id = fix['fixture']['id']

        home = fix['teams']['home']['id']
        away = fix['teams']['away']['id']
        homeGoals = fix['goals']['home']
        awayGoals = fix['goals']['away']
        status = fix['fixture']['status']['long'] #'Match Finished'
        all_fix.append((id, home, away, homeGoals, awayGoals, status))

    return all_fix


#returns list of the two teams playing
# api calls 1-10 REQUESTS
#@param: format is 'name' or 'id' 
def teams_in_match(fixture_id, format):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {"id":fixture_id}

    response = proper_request("GET", url, params = querystring, headers=HEADERS)
    return [response.json()['response'][0]['teams'][location][format] for location in ['home', 'away']]


# returns dictionary of the 8 stats, Sh, Sa, ST, F, C dict (8 items)
# api calls 1-10 REQUESTS
def get_match_stats(fixture_id):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics"
    querystring = {"fixture":fixture_id}

    response = proper_request("GET", url, headers=HEADERS, params=querystring, max_requests=10)
    teams = response.json()['response']
    info = {}
    def findstat(clovara, stattype):
        for stat in clovara['statistics']:
            if stat['type']==stattype:
                val = stat['value']
                return (0 if val is None else val)
        raise Exception("couldn't find the appropriate stat type")

    info['STh'] = findstat(teams[0], "Shots on Goal")
    info['Sh'] =  findstat(teams[0],"Total Shots")
    info['Fh'] =  findstat(teams[0],"Fouls")
    info['Ch'] =  findstat(teams[0],"Corner Kicks")
    info['STa'] =  findstat(teams[1],"Shots on Goal")
    info['Sa'] = findstat(teams[1],"Total Shots")
    info['Fa'] = findstat(teams[1],"Fouls")
    info['Ca'] = findstat(teams[1],"Corner Kicks")
    return info

# returns dictionary with keys as fixture id and tuple of oddsH, oddsD, oddsA as val
# api calls 1-10 REQUESTS
def get_bet365_odds(league_id):
    url = "https://api-football-v1.p.rapidapi.com/v3/odds"
    querystring = {"league":league_id,"season": INT_SEASON_START}

    response = proper_request("GET", url, headers=HEADERS, params=querystring, max_requests=10)
    fixtures = response.json()['response']
    all_fix = {}
    for fix in fixtures:
        try:
            bookies = fix['bookmakers']
            index = 0
            for n in range(len(bookies)):
                if bookies[n]['name']=='Bet365':
                    index=n
                    break
            bet365= bookies[index]['bets'][0]['values'] #match winner
            odds = [0,0,0]
            for odd in bet365:
                if odd['value'] == 'Home':
                    odds[0] = odd['odd']
                elif odd['value'] == 'Draw':
                    odds[1] = odd['odd']
                elif odd['value'] == 'Away':
                    odds[2] = odd['odd']
            fix_id = fix['fixture']['id']
            all_fix[fix_id] = tuple(odds)
        except:
            print('Tried to get bet365 data for singular fixture but got error')
            continue
    return all_fix


# takes in list of ids (int), 
# returns dictionary with keys as fixture id and tuple of oddsH, oddsD, oddsA as val
# api calls 1-10 REQUESTS
def get_bet365_odds_by_fixtures(fixture_id_list):

    all_fix = {}
    for fixture_id in fixture_id_list:
        url = "https://api-football-v1.p.rapidapi.com/v3/odds"
        querystring = {"fixture":fixture_id, "season":INT_SEASON_START}

        response = proper_request("GET", url, params=querystring, headers=HEADERS)
        fix_all = response.json()['response']
        try:
            fix = fix_all[0]
        except:
            print('some sort of failure to get this fixtures odds')
            print(fixture_id, ': ', response.json()['response'])
            continue

        try:
            bookies = fix['bookmakers']
            index = 0
            for n in range(len(bookies)):
                if bookies[n]['name']=='Bet365':
                    index=n
                    break
            bet365= bookies[index]['bets'][0]['values'] #match winner
            odds = [0,0,0]
            for odd in bet365:
                if odd['value'] == 'Home':
                    odds[0] = odd['odd']
                elif odd['value'] == 'Draw':
                    odds[1] = odd['odd']
                elif odd['value'] == 'Away':
                    odds[2] = odd['odd']
            fix_id = fix['fixture']['id']
            all_fix[fix_id] = tuple(odds)
        except:
            print('Tried to get bet365 data for singular fixture but got error')

    return all_fix


# return: dictionary id --> name ***the manchesters are full name, so we will match some other way
def team_id_converter_api(league_id):
    url = "https://api-football-v1.p.rapidapi.com/v3/standings"
    querystring = {"season":INT_SEASON_START,"league":league_id}

    response = proper_request("GET", url, headers=HEADERS, params=querystring, max_requests=10)
    all_teams = {val['team']['id']:val['team']['name'] for val in response.json()['response'][0]['league']['standings'][0]}
    return all_teams

"""Get Match Stats to Debug
league_id = get_premier_league_id(2021)
fix_ids = get_fixture_ids(league_id, '2021-08-23')
print(fix_ids)
for id in [x[0] for x in fix_ids]:
    print(id)
    stats = get_match_stats(id)
    print(stats)
"""