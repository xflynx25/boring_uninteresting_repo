#2019-08-10T11:30:00Z
def difference_in_days(start_day, end_day):
    root_year, root_month, root_day = start_day
    year, month, day = end_day

    def is_leap_year(year):
        if year % 400 == 0:
            return True
        elif year % 100 == 0:
            return False
        elif year % 4 == 0:
            return True
        else: 
            return False
    # Check Leap Year
    leap_year = is_leap_year(root_year + 1)
    if leap_year:
        feb = 29
    else:
        feb = 28

    # month dict
    days_in_month = {
        1: 31,
        2: feb,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 31,
        10: 31,
        11: 30,
        12: 31,
    } 
    #convert dates into days based on day0
    difference = 0
    if year < root_year or (year==root_year and (month<root_month or (month==root_month and day < root_day))):
        raise Exception("start is after end")

    for this_year in range(root_year, year+1):
        if this_year not in (root_year, year): #go through full year 
            difference += 365 + [1 if is_leap_year(this_year) else 0][0]
        else:
            if this_year == year: # go until the stop date in the year 
                if year != root_year:
                    root_month, root_day = 1,0 #if we are coming from previous year 

                if month == root_month:
                    difference += day - root_day
                else:
                    difference += days_in_month[root_month] - root_day
                    for m in range(root_month + 1, month):
                        difference += days_in_month[m]
                    difference += day

            elif this_year == root_year: #go to end of year 
                difference += days_in_month[root_month] - root_day
                for m in range(root_month + 1, 13):
                    difference += days_in_month[m]
                #for m in range(1, month):
                #    difference += days_in_month[m]
                #difference += day
 
    
    return difference

# h, m, s  # DOES NOT ACCOUNT FOR SAME EXACT TIME
def which_time_comes_first(a,b):
    if a[0] != b[0]: 
        first = b[0] < a[0] 
    else:
        if a[1] != b[1]: 
            first = b[1] < a[1] 
        else:
            if a[2] != b[2]:
                first = b[2] < a[2]
            else:
                return -1 

    return [1 if first else 0][0]


'''returns dataframe without any columns containing the str in combos'''
'''now with support for series'''
def drop_columns_containing(patterns, df):
    drop_indices = set()
    is_frame = 2 == len(df.shape)

    if is_frame:
        cols = df.columns
    else:
        cols = df.index
    for pattern in patterns:
        if is_frame:
            truth_table = df.columns.str.contains(pattern)
        else:
            truth_table = df.index.str.contains(pattern)
        for index in range(len(cols)):
            if truth_table[index]:
                drop_indices.add(index)
    
    drop_cols = []
    for index in drop_indices:
        drop_cols.append(cols[index])
    if is_frame:
        return df.drop(drop_cols, axis=1)
    else:
        return df.drop(drop_cols)

'''returns df with columns having some pattern in patterns'''
'''now with support for series'''
def get_columns_containing(patterns, df):
    is_frame = 2 == len(df.shape)

    if is_frame:
        cols = df.columns
    else:
        cols = df.index
    truth_table = [False for x in cols]
    for pattern in patterns:
        if is_frame:
            this_pattern = df.columns.str.contains(pattern)
        else: 
            this_pattern = df.index.str.contains(pattern)
        truth_table = this_pattern | truth_table

    if is_frame:
        return df.iloc[:, truth_table]
    else:
        return df.loc[truth_table]
    