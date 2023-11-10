import numpy as np 
from pyfit_utilities import round_to_sig_figs
import time

# Define the number of significant figures for rounding
ROUND_SIG_FIGS = 4




class ModelComparison:
    def __init__(self, models, datasets, loss_func, model_labels=None, dataset_labels=None, starred_pairs=None, **meta_kwargs):
        self.models = models
        self.datasets = datasets
        self.loss_func = loss_func
        self.model_labels = model_labels or [str(i) for i in range(len(models))]
        self.dataset_labels = dataset_labels or [str(i) for i in range(len(datasets))]
        self.starred_pairs = starred_pairs or []
        self.meta_kwargs = meta_kwargs
        self.results = None
        self.starred_results = {}  # To store predictions and actuals for starred pairs

    def run(self):
        results = []
        total_models = len(self.models)
        total_datasets = len(self.datasets)
        for i, model in enumerate(self.models):
            print(f"Processing Model {i+1}/{total_models}: {self.model_labels[i]}")
            model_results = []
            for j, dataset in enumerate(self.datasets):
                print(f"   Testing on Dataset {j+1}/{total_datasets}: {self.dataset_labels[j]}")
                X_test, y_test = dataset
                start_time = time.time()  # Capture the time before prediction
                predictions = model.predict(X_test)
                end_time = time.time()  # Capture the time after prediction

                # Calculate the prediction time
                prediction_time = end_time - start_time
                loss_values = [self.loss_func(y_true, y_pred) for y_true, y_pred in zip(y_test, predictions)]
                stats = {
                    "avg_loss": round_to_sig_figs(sum(loss_values) / len(loss_values), ROUND_SIG_FIGS),
                    "worst_loss": round_to_sig_figs(max(loss_values), ROUND_SIG_FIGS),
                    "std_loss": round_to_sig_figs(np.std(loss_values), ROUND_SIG_FIGS),  # Assuming numpy is used
                    "prediction_time": round_to_sig_figs(prediction_time, ROUND_SIG_FIGS),
                    "all_loss_values": [round_to_sig_figs(val, ROUND_SIG_FIGS) for val in loss_values]
                }
                model_results.append(stats)
                
                # If the current pair is starred, save the predictions and actuals
                if (self.model_labels[i], self.dataset_labels[j]) in self.starred_pairs:
                    key = (self.model_labels[i], self.dataset_labels[j])
                    self.starred_results[key] = (predictions, y_test)
            results.append(model_results)
        self.results = StatisticalDataset(results)


class StatisticalDataset:
    def __init__(self, data):
        self.data = data  # 3D list or array

from pyfit_visualizations import pred_actual_2d  # Assuming Visualization is a separate module

class Visualization:
    def __init__(self, comparison_obj):
        self.dataset = comparison_obj.results
        self.starred_results = comparison_obj.starred_results

    def plot_starred_pairs(self):
        for key, (predictions, actuals) in self.starred_results.items():
            model_label, dataset_label = key
            title = f"Predicted vs Actual for Model: {model_label}, Dataset: {dataset_label}"
            pred_actual_2d(predictions, actuals, title=title)
