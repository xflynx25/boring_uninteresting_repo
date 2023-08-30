import numpy as np 

# Function to round to a specified number of significant figures
def round_to_sig_figs(num, sig_figs):
    if num != 0:
        return round(num, -int(np.floor(np.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # return 0 if the number is 0