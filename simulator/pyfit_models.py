from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def get_neural_network(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', max_iter=1000, early_stopping=False, **kwargs):
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, max_iter=max_iter, early_stopping=early_stopping, **kwargs)

def get_random_forest(n_estimators=10, **kwargs):
    return RandomForestClassifier(n_estimators=n_estimators, **kwargs)

def get_logistic_regression(regularization='none', **kwargs):
    if regularization == 'none':
        return LogisticRegression(penalty=None, max_iter=1000, **kwargs)
    elif regularization == 'l1':
        print('bro stop trying lasso it takes too long')
        return LogisticRegression(penalty='l1', solver='saga', max_iter=1000, **kwargs)
    elif regularization == 'l2':
        return LogisticRegression(penalty='l2', max_iter=1000, **kwargs)
    else:
        raise ValueError(f"Invalid regularization type: {regularization}")

