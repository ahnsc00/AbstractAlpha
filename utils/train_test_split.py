def split_data(data, train_ratio=0.8):
    """
    Splits the data into training and testing sets based on the given train-test ratio.

    Parameters:
        data (DataFrame or array-like): The dataset to split.
        train_ratio (float): The proportion of the data to use for training (default is 0.9).

    Returns:
        train_data, test_data: The training and testing sets.
    """
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data