class Oracle:
    """Oracle

    This class has access to the true labels of the data and can be used to query the true labels of the data.

    Args:
        X_u (np.ndarray): The unlabeled data.
        y_u (np.ndarray): The true labels of the unlabeled data.

    Attributes:
        X_u (np.ndarray): The unlabeled data.
        y_u (np.ndarray): The true labels of the unlabeled data.
    """

    def __init__(self, X_u=None, y_u=None):
        self.X_u = X_u
        self.y_u = y_u

    def set_data(self, X_u, y_u):
        """Initializes the data."""
        self.X_u = X_u
        self.y_u = y_u

    def query(self, indices_to_label):
        """Queries the true labels of the data.

        Parameters:
            indices_to_label (np.ndarray): The indices of the instances to query.
        Returns:
            np.ndarray: The true labels of the queried instances.
        """
        return self.y_u[indices_to_label]
