

class Oracle:

    def __init__(self, X_u=None, y_u=None):
        self.X_u = X_u
        self.y_u = y_u

    def set_data(self, X_u, y_u):
        self.X_u = X_u
        self.y_u = y_u

    def query(self, indices_to_label):
        return self.y_u[indices_to_label]
