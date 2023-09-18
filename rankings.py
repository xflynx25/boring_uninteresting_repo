"""
I want to have a sense of what ranking people would have gotten, in a specific year, given a number of points. 
This is to evaluate my bots performance. 
If I could somehow get number of players (I can estimate), then this could be a better metric than points for RL 
So I will manually grab user data, and graph to visualize, and see if I can fit an interpolation function
"""

from constants import DROPBOX_PATH
from general_helpers import safe_read_csv, drop_columns_containing, safe_to_csv
import matplotlib.pyplot as plt 
import numpy as np
from scipy.optimize import curve_fit
print('start')
inpath = DROPBOX_PATH + "Human_Seasons/season_scores.csv"
savepath = DROPBOX_PATH + "Human_Seasons/season_scores_cleaned.csv"
df = safe_read_csv(inpath)
df = drop_columns_containing("Unn", df)
safe_to_csv(df, savepath)
outpath = DROPBOX_PATH + "Human_Seasons/season_scores_plot.png"
print('read')

print([col for col in df.columns])
unique_years = set(col[:4] for col in df.columns[1:])
print(unique_years)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

def negative_logistic(x, a, b, c, d):
    return a - b / (1 + np.exp(c * (x - d)))

# Define the polynomial function
def func(x, a, b, c, d):
    return a * x + b * x**2 + c/x + d

# Iterate over unique years
for year in unique_years:
    print(f'year: {year}')
    score_column = f"{year}-score"
    rank_column = f"{year}-rank"

    # Drop NaN values for the current year's columns
    filtered_df = df[[score_column, rank_column]].dropna()

    x_data = filtered_df[score_column].to_numpy()
    y_data = filtered_df[rank_column].to_numpy()

    # Fit the polynomial
    params, params_covariance = curve_fit(negative_logistic, x_data, y_data, maxfev=5000)

    # Calculate R^2
    residuals = y_data - negative_logistic(x_data, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Plot scatter plot
    ax.scatter(x_data, y_data, label=f"{year}")

    # Plot the fit
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    ax.plot(x_fit, negative_logistic(x_fit, *params), '-')

    # Update the label with fit parameters and R^2
    ax.legend([f"{year}, a={params[0]:.2f}, b={params[1]:.2f}, c={params[2]:.2f}, d={params[3]:.2f}, $R^2$={r_squared:.2f}"])

# Set labels and title
ax.set_xlabel('Score')
ax.set_ylabel('Rank')
ax.set_title('Scatter plot of Score vs. Rank for different years with polynomial fit')

print('saving')

# Optionally, save the plot to the desired location
plt.savefig(outpath)

# Display the plot
plt.show()

