import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

def calc_rmse(*args):
    '''Root Mean Squared Error'''
    mse = mean_squared_error(*args)
    return np.sqrt(mse)


def calc_mape(y_true, y_pred):
    '''Mean Absolute Percentage Error'''
    rel_error = (y_true - y_pred) / y_true
    return 100 * np.mean(np.abs(rel_error))


def eval_multi(y_true, y_pred_multi, metric, scaler):
    results = []
    for i in range(y_pred_multi.shape[1]):
        y_pred = y_pred_multi[:, i]
        if i > 0:
            y_pred = y_pred[:-i]

        y_true_inv = scaler.inverse_transform(y_true[i:])
        y_pred_inv = scaler.inverse_transform(y_pred)

        res = metric(y_true_inv, y_pred_inv)
        results.append(res)

    return results


def report_metrics(metrics, names):
    report = pd.DataFrame(
        data=metrics,
        columns=names
    )

    n_steps = metrics.shape[0]
    report['Step'] = np.arange(1, n_steps+1)
    report.set_index('Step', inplace=True)

    return report


def visualize_pred_ext(y_test, y_pred_multi, title, y, split_year, scaler, y_train=None, limit=-1):
    for i in range(y_pred_multi.shape[1]):
        end_idx = i+1
        visualize_pred(y_test, y_pred_multi[:, i:end_idx], '{} ({} steps)'.format(title, end_idx), y, split_year, scaler, legend=False)


def visualize_pred(y_test, y_pred_multi, title, y, split_year, scaler, y_train=None, limit=-1, legend=True):
    fig, ax = plt.subplots(figsize=(20, 10))

    if y_train is not None:
        # Training set
        ax.plot(y[:split_year].index, y_train, label='Train')

    # Test set truth
    y_test_inv = scaler.inverse_transform(y_test)
    ax.plot(y[split_year:].index[:limit], y_test_inv[:limit], label='Test truth')

    # Test predictions
    for i in range(y_pred_multi.shape[1]):
        y_pred = y_pred_multi[:, i]
        if i > 0:
            y_pred = y_pred[:-i]

        y_pred_inv = scaler.inverse_transform(y_pred)
        rmse = calc_rmse(y_test_inv[i:], y_pred_inv)

        indices = y[split_year:].index[i:]
        ax.plot(indices[:limit], y_pred_inv[:limit],
                label='{}-step (RMSE={:0.4f})'.format(i+1, rmse) )
    if legend:
        ax.legend()
    ax.set_title(title)


def recursive_forecast(model, X_test, y_test, n_steps=1):
    # Make an accumulator for predictions
    predictions = np.zeros(shape=(y_test.shape[0], n_steps))
    predictions[:] = np.nan

    X_test_step = X_test.copy()

    for i in range(n_steps):
        predictions[:, i] = model.predict(X_test_step)
        X_test_step = np.concatenate((X_test_step[:, 1:], predictions[:, i:i+1]), axis=1)

    return predictions


def evaluate_model(model, dataset, scaler, n_steps=1):
    # Unpack the dataset
    X_train, y_train, X_test, y_test = dataset

    # Fit model
    model.fit(X_train, y_train)

    # Make recursive multi-step predictions
    y_pred_multi = recursive_forecast(model, X_test, y_test, n_steps)

    # Evaluate
    svr_rmse = eval_multi(y_test, y_pred_multi, calc_rmse, scaler)
    svr_mape = eval_multi(y_test, y_pred_multi, calc_mape, scaler)

    # Report the metrics
    metrics = np.array([svr_rmse, svr_mape]).T
    summary = report_metrics(metrics, ['RMSE', 'MAPE'])

    return summary, y_pred_multi


def frame_supervised_data(data, n_features=1, n_forecast=1, forecast_columns=None, multi_steps=False):
    ''' A function that converts time series data
    into supervised learning data (inputs and outputs)
    '''

    # Extract the number of variables
    n_vars = 1
    # Check for multivariate data
    if data.ndim > 1:
        n_vars = data.shape[1]

    # Create (copy) a dataframe of the data
    df_temp = data.copy()

    # Extract the column names if dataframe
    #column_names = ['var{0}'.format(c) for c in range(n_vars)]
    #if type(df_temp) is pd.core.frame.DataFrame:
    column_names = data.columns

    framed_data = []
    framed_feature_columns = []

    # Create the features sequence by shifting timestep backward (t-n_features, ..., t-2, t-1)
    for i in range(n_features, 0, -1):
        shifted_data = df_temp.shift(i)
        framed_data.append(shifted_data)
        framed_feature_columns += ['{0}(t-{1})'.format(c, i) for c in column_names]


    # If output columns are specified, shift only them
    forecast_columns = forecast_columns or column_names
    framed_forecast_columns = []
    # Create the forecast sequence by shifting timestemp forward (t, t+1, t+2, ... t+n_forecast)
    for i in range(0, n_forecast):
        shifted_data = df_temp[forecast_columns].shift(-i)

        if multi_steps or (i == n_forecast-1):
            framed_data.append(shifted_data)

            # Make t+0 just t
            if i == 0:
                framed_forecast_columns += ['{0}(t)'.format(c) for c in forecast_columns]
            else:
                framed_forecast_columns += ['{0}(t+{1})'.format(c, i) for c in forecast_columns]

    # Combine the columns
    df_framed_data = pd.concat(framed_data, axis=1)
    df_framed_data.columns = framed_feature_columns + framed_forecast_columns

    return df_framed_data.dropna(), framed_forecast_columns
