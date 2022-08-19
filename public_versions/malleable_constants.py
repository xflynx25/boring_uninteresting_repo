COMPUTER_USERNAME = ""
C_ENTRY = "" # for automation we run on windows. 

def change_computer_username(username):
    global COMPUTER_USERNAME
    COMPUTER_USERNAME = username


def change_c_entry(val):
    global C_ENTRY
    C_ENTRY = val