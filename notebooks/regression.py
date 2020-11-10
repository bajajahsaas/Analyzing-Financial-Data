# Import python modules
import numpy as np
from sklearn.model_selection import KFold
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# Compute MAE
def compute_error(y_hat, y):
    # mean absolute error
    return np.abs(y_hat - y).mean()

# Compute MSE
def compute_MSE(y, y_hat):
    # mean squared error
    return np.mean(np.power(y - y_hat, 2))

############################################################################

# Perform k-fold cross-validation

def get_cross_val_score(model, data_num, hyperparam, train_x, train_y):
    start_time = int(round(time.time() * 1000))
    kf = KFold(n_splits=5)
    kf.get_n_splits(train_x)
    errors = []
    for train_index, test_index in kf.split(train_x):
        X_train, X_val = train_x[train_index], train_x[test_index]
        y_train, y_val = train_y[train_index], train_y[test_index]
        X_train, y_train = shuffle(X_train, y_train)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_val)
        error = compute_error(y_hat, y_val)
        errors.append(error)

    mean_error = np.mean(errors)
    end_time = int(round(time.time() * 1000))
    # print("Mean error for data {0} and hyperparameter {1}: {2:.4f}".format(data_num, hyperparam, mean_error))
    # print("--- %s milli-seconds ---" % (end_time - start_time))
    return mean_error, end_time - start_time


# fit Linear regression model with cross validation

def linear_regression(data_num, train_x, train_y, test_x):
    print('Running linear regression for data type:', data_num)
    
    model = LinearRegression()
    this_error, this_time = get_cross_val_score(model, data_num, None, train_x, train_y)

    print("After cross-val for data {0} mean Error: {1:0.4f}".format(data_num, this_error))
    print("Training model on full data")

    model = LinearRegression()
    model.fit(train_x, train_y)
    y_hat = model.predict(test_x)
    return y_hat, model


# fit decision tree regression model with cross validation

def decision_trees(data_num, train_x, train_y, test_x):
    print('Running decision tree regressor for data type:', data_num)
    
    depths = [3, 6, 9, 12, 15, 20, 25, 30, 35, 40]  # tune over max_depth
    best_depth = -1
    min_error = float('inf')
    times = []
    for depth in depths:
        model = DecisionTreeRegressor(criterion='mae', max_depth=depth)
        this_error, this_time = get_cross_val_score(model, data_num, depth, train_x, train_y)
        times.append(this_time)
        if this_error < min_error:
            min_error = this_error
            best_depth = depth

    print("After cross-val, best depth for data {0} is {1} with mean Error: {2:0.4f}".format(data_num, best_depth,
                                                                                             min_error))
    fig, ax = plt.subplots()
    ax.plot(depths, times)
    ax.set(xlabel='maximum depth', ylabel='time (ms)',
           title='Run time of decision tree regressor for varying maximum depth (data {0})'.format(data_num))
    fig.savefig('plots/decisiontree_data{0}.png'.format(data_num))
    # plt.show()

    print("Training model on full data")

    model = DecisionTreeRegressor(criterion='mae', max_depth=best_depth)
    model.fit(train_x, train_y)
    y_hat = model.predict(test_x)
    return y_hat, model


# fit KNN regression model with cross validation

def k_nearest_neighbors(data_num, train_x, train_y, test_x):
    print('Running k-nearest neighbor regressor for data type:', data_num)
    neighbors = [3, 5, 10, 20, 25]  # tune over neighbor count
    best_k = -1
    min_error = float('inf')
    times = []
    for neigh in neighbors:
        model = KNeighborsRegressor(n_neighbors=neigh)
        this_error, this_time = get_cross_val_score(model, data_num, neigh, train_x, train_y)
        times.append(this_time)
        if this_error < min_error:
            min_error = this_error
            best_k = neigh

    print("After cross-val, best neighbors for data {0} is {1} with mean Error: {2:0.4f}".format(data_num, best_k,
                                                                                                 min_error))
    fig, ax = plt.subplots()
    ax.plot(neighbors, times)
    ax.set(xlabel='neighbor count (k) ', ylabel='time (ms)',
           title='Run time of KNN regressor for varying k (data {0})'.format(data_num))
    fig.savefig('plots/KNN_data{0}.png'.format(data_num))
    # plt.show()
    print("Training model on full data")

    model = KNeighborsRegressor(n_neighbors=best_k)
    model.fit(train_x, train_y)
    y_hat = model.predict(test_x)
    return y_hat, model


# fit ridge and lasso regression models with cross validation

def ridge_and_lasso(data_num, train_x, train_y, test_x):

    alphas = [1e-6, 1e-4, 1e-2, 1, 10]  # tune over regularization constant

    methods = ['ridge', 'lasso']
    best_method = None
    best_alpha = -1
    min_error = float('inf')
    times_r = []
    times_l = []
    for method in methods:
        print('Running {0} regressors for data type: {1}'.format(method, data_num))
        for alpha in alphas:
            if method == "ridge":
                model = linear_model.Ridge(alpha=alpha)
            elif method == 'lasso':
                model = linear_model.Lasso(alpha=alpha)

            this_error, this_time = get_cross_val_score(model, data_num, alpha, train_x, train_y)

            if method == "ridge":
                times_r.append(this_time)
            elif method == 'lasso':
                times_l.append(this_time)

            if this_error < min_error:
                min_error = this_error
                best_alpha = alpha
                best_method = method

    print("After cross-val, best method for data {0} is {1} with alpha = {2} and mean Error: {3:0.4f}".format(data_num,
                                                                                                              best_method,
                                                                                                              best_alpha,
                                                                                                              min_error))

    fig, ax = plt.subplots()
    ax.plot(alphas, times_r, label='ridge')
    ax.plot(alphas, times_l, label='lasso')
    ax.set(xlabel='alpha', ylabel='time (ms)',
           title='Run time of ridge and lasso regressor for varying alpha (data {0})'.format(data_num))
    fig.savefig('plots/ridgelasso_data{0}.png'.format(data_num))
    ax.legend()
    # plt.show()
    print("Training model on full data")

    if best_method == "ridge":
        model = linear_model.Ridge(alpha=best_alpha)
    elif best_method == 'lasso':
        model = linear_model.Lasso(alpha=best_alpha)

    model.fit(train_x, train_y)
    y_hat = model.predict(test_x)
    return y_hat, model


