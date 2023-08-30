# temporarily do this here, should have classes related to visualizing .. 

import matplotlib.pyplot as plt
import numpy as np

def pred_actual_2d(predictions, actuals, title=None, filename=None):
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title or "Predicted vs Actual Values")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_error_bars(dataset, model_labels, dataset_labels, filename=None):
    means = []
    stds = []
    labels = []

    for i, model_result in enumerate(dataset.data):
        for j, stats in enumerate(model_result):
            means.append(stats["avg_loss"])
            stds.append(stats["std_loss"])
            labels.append(f"{model_labels[i]}-{dataset_labels[j]}")

    # Sorting based on means
    sorted_indices = np.argsort(means)
    sorted_means = np.array(means)[sorted_indices]
    sorted_stds = np.array(stds)[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]

    plt.bar(range(len(sorted_means)), sorted_means, yerr=sorted_stds, align='center', alpha=0.7, color='b', capsize=10)
    plt.xticks(range(len(sorted_means)), sorted_labels, rotation=45)
    plt.ylabel('Average Loss')
    plt.title('Model Performance')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


# #make so still useful if outliers
# because right now, if we have one really bad model, you will not be able to see the difference, 
# what we would like is for us to do the thing with the graph which does a break with the two diagonal lines on axis
# which means that we skip a lot of values 
def plot_error_scatter(dataset, model_labels, dataset_labels, filename=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    means = []
    labels = []

    for i, model_result in enumerate(dataset.data):
        for j, stats in enumerate(model_result):
            means.append(stats["avg_loss"])
            labels.append(f"{model_labels[i]}-{dataset_labels[j]}")

    sorted_indices = np.argsort(means)
    sorted_means = np.array(means)[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]

    # plot the same data on both axes
    ax1.scatter(range(len(sorted_means)), sorted_means, alpha=0.7, color='b')
    ax2.scatter(range(len(sorted_means)), sorted_means, alpha=0.7, color='b')

    ax1.set_ylim(0.8, 1.)  # outliers only
    ax2.set_ylim(0, .5)  # most of the data

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')
    ax2.xaxis.tick_bottom()

    plt.xticks(range(len(sorted_means)), sorted_labels, rotation=45)
    plt.ylabel('Average Loss')
    plt.title('Model Performance')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


# same problem as above with the big outliers, here it is happening also from the outliers within a single sample,
#  so it will be good to do this plot without the outliers included, so that we have better granularity in vieewing it. 
def plot_box_and_whisker(dataset, model_labels, dataset_labels, filename=None):
    all_data = []
    labels = []

    for i, model_result in enumerate(dataset.data):
        for j, stats in enumerate(model_result):
            # Here, assuming stats also has "all_loss_values" which is a list of all loss values
            all_data.append(stats["all_loss_values"])
            labels.append(f"{model_labels[i]}-{dataset_labels[j]}")

    plt.boxplot(all_data, showfliers=False)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.ylabel('Loss Values')
    plt.title('Model Performance')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


# we want to also have a plot timings function for the training, not just the predictions. 
def plot_timings(dataset, model_labels, dataset_labels, filename=None):
    timings = []
    labels = []

    for i, model_result in enumerate(dataset.data):
        for j, stats in enumerate(model_result):
            timings.append(stats["prediction_time"])
            labels.append(f"{model_labels[i]}-{dataset_labels[j]}")

    plt.bar(range(len(timings)), timings, align='center', alpha=0.7, color='b')
    plt.xticks(range(len(timings)), labels, rotation=45)
    plt.ylabel('Prediction Time (s)')
    plt.title('Prediction Timings')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

# Function to plot training times
def plot_training_timings(training_times, filename=None):
    timings = [x[1] for x in training_times]
    labels = [x[0] for x in training_times]

    plt.bar(range(len(timings)), timings, align='center', alpha=0.7, color='b')
    plt.xticks(range(len(timings)), labels, rotation=45)
    plt.ylabel('Training Time (s)')
    plt.title('Training Timings')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()