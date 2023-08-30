# import rl simulator and visualizations from pyfit
# set parameters from reward_func, state_evolution, model, and stuff from config
# in config, we have things like the state size, initial state, action space size, and hyperparameters from the simulation 
# hyperparameters can be things like num_rounds, 

# set up the simulator object 
# call simulator.step. Can access class things like reward_list, current_state, action_list, 
# maybe we can have a visualizer function where you can pass in a mask for columns which are the same, and also names, for example things that are 2 transfers, and then be able to plot things from there

# by the end, you will be able to plot rewards, and you can get meta information from the action list. 
# instead, you can call it many times, and it will record only the last lists, and maintain a throughout simulation histogram 
# you can also import compare models class, which will run this many times, given different model imports, 
    # and give you access to good plotting functions 

# lastly, we need to put something in constants which is the root path to this dir, so we can properly import in this subdir 

from pyfit_visualizations import plot_error_bars, plot_timings, plot_training_timings
from pyfit_visualizations import plot_error_scatter, plot_box_and_whisker
from pyfit_datasets import get_mnist_dataset, get_random_classification_dataset
from pyfit_models import get_random_forest, get_logistic_regression, get_neural_network  # Add get_neural_network
from pyfit_utilities import round_to_sig_figs

from pyfit_supervised_learning import ModelComparison, Visualization  # Assuming the ModelComparison class and others are in a module named model_comparison_module

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import time

import logging

ROUND_SIG_FIGS = 4

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),  # Log to this file
                        logging.StreamHandler()  # Log to console
                    ])

# Now you can use the logging module to log messages
logging.info("This is an info message.")
logging.error("This is an error message.")


def train_and_evaluate(model, X_train, y_train, X_val=None, y_val=None, cv=5):
    # Fit the model
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    # Training accuracy
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    
    metrics = {
        "Training Accuracy": train_accuracy,
        "Training Time": round_to_sig_figs(training_time, ROUND_SIG_FIGS)
    }
    
    # If validation data is provided
    if X_val is not None and y_val is not None:
        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_report = classification_report(y_val, val_predictions, output_dict=True)
        
        metrics["Validation Accuracy"] = val_accuracy
        metrics["Validation Report"] = val_report

    # If no validation data is provided, use cross-validation
    else:
        cross_val_accuracies = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        metrics["Cross-Validation Mean Accuracy"] = cross_val_accuracies.mean()
        metrics["Cross-Validation Std Accuracy"] = cross_val_accuracies.std()
        
    for key, value in metrics.items():
        if isinstance(value, dict):  # Handle the classification report separately
            logging.info(f"{key}:")
            for sub_key, sub_val in value.items():
                logging.info(f"    {sub_key}: {round_to_sig_figs(sub_val, ROUND_SIG_FIGS)}")
        else:
            logging.info(f"{key}: {round_to_sig_figs(value, ROUND_SIG_FIGS)}")
    logging.info('\n')
    
    return model, training_time




def zero_one_loss(y_true, y_pred):
    return 1 if y_true != y_pred else 0

if __name__ == "__main__":
    training_times = []
    # Get datasets
    mnist_dataset = get_mnist_dataset()
    random_dataset = get_random_classification_dataset()
    
    # Get models
    rf5 = get_random_forest(n_estimators=5)
    rf20 = get_random_forest(n_estimators=20)
    rf100 = get_random_forest(n_estimators=100)
    rf200 = get_random_forest(n_estimators=200)
    lr_none = get_logistic_regression(regularization='none')
    lr_ridge = get_logistic_regression(regularization='l2')
    nn1 = get_neural_network(hidden_layer_sizes=(200,), max_iter=1000)
    nn2 = get_neural_network(hidden_layer_sizes=(50, 50), max_iter=1000)
    nn3 = get_neural_network(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=1000)
    nn4 = get_neural_network(hidden_layer_sizes=(100, 50, 50), max_iter=1000)

    # Train the models
    
    # can we find a way to do a plot timing on the training times? 
    rf5, t = train_and_evaluate(rf5, mnist_dataset.train, mnist_dataset.target_train)
    training_times.append(('Random Forest 5', t))
    rf20, t = train_and_evaluate(rf20, mnist_dataset.train, mnist_dataset.target_train)
    training_times.append(('Random Forest 20', t))
    rf100, t = train_and_evaluate(rf100, mnist_dataset.train, mnist_dataset.target_train)
    training_times.append(('Random Forest 100', t))
    rf200, t = train_and_evaluate(rf200, mnist_dataset.train, mnist_dataset.target_train)
    training_times.append(('Random Forest 200', t))
    lr_none, t = train_and_evaluate(lr_none, mnist_dataset.train, mnist_dataset.target_train)
    training_times.append(('lr none', t))
    lr_ridge, t = train_and_evaluate(lr_ridge, mnist_dataset.train, mnist_dataset.target_train)
    training_times.append(('lr ridge', t))
    nn1, t = train_and_evaluate(nn1, mnist_dataset.train, mnist_dataset.target_train)
    training_times.append(('nn1', t))
    nn2, t = train_and_evaluate(nn2, mnist_dataset.train, mnist_dataset.target_train)
    training_times.append(('nn2', t))
    nn3, t = train_and_evaluate(nn3, mnist_dataset.train, mnist_dataset.target_train)
    training_times.append(('nn3', t))
    nn4, t = train_and_evaluate(nn4, mnist_dataset.train, mnist_dataset.target_train)
    training_times.append(('nn4', t))

    # Create model comparison object
    model_labels = ['Random Forest 5', 'Random Forest 20','Random Forest 100','Random Forest 200','lr none','lr ridge', 'nn1','nn2','nn3','nn4']
    models = [rf5, rf20, rf100, rf200, lr_none, lr_ridge, nn1,nn2,nn3,nn4]
    datasets = [(mnist_dataset.test, mnist_dataset.target_test)]#, (random_dataset.test, random_dataset.target_test)]
    
    dataset_labels = ['MNIST']#, 'Random Data']
    starred_pairs = [('MNIST')]#('Random Forest', 'MNIST')]  # Example of starring Random Forest on MNIST

    # we can try to do the test on only mnist to see better visual
    
    comparison = ModelComparison(models, datasets, loss_func=zero_one_loss, model_labels=model_labels, dataset_labels=dataset_labels, starred_pairs=starred_pairs)
    comparison.run()
    
    # Visualization
    vis = Visualization(comparison)
    vis.plot_starred_pairs()

    # Plot error bars
    # not that useful, std of error not too interesting 
    # plot_error_bars(comparison.results, model_labels, dataset_labels, filename="error_bars.png")

    # Plotting using the new functions
    plot_error_scatter(comparison.results, model_labels, dataset_labels, filename="error_scatter.png") 
    
    #not useful because the outliers make the grid take too much space, maybe not include the outliers 
    plot_box_and_whisker(comparison.results, model_labels, dataset_labels, filename="box_and_whisker.png") 
    plot_timings(comparison.results, model_labels, dataset_labels, filename="prediction_timings.png")


    # Plotting the training times
    plot_training_timings(training_times, filename="training_timings.png")


