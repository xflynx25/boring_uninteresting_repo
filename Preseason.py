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

'''