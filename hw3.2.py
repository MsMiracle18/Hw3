def compute_cost(X, y, w):
    m = len(y)
    predictions = hypothesis(X, w)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost
