#!/usr/bin/python


def outlierCleaner(predictions, X_train_scaled, y_train):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    residuals = y_train - predictions
    data_with_errors = [(X_train_scaled, y_train, error) for X_train_scaled, y_train, error in zip(X_train_scaled, y_train, residuals)]
    data_with_errors.sort(key=lambda x: abs(x[2]))
    cleaned_data = data_with_errors[:int(1.4 * len(data_with_errors))]
    return cleaned_data

