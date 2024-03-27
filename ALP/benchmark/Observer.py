from abc import ABC, abstractmethod


class Observer(ABC):

    def __init__(self):
        return

    @abstractmethod
    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug):
        pass

    @abstractmethod
    def observe_model(self, iteration, model):
        pass


class TestPerformanceObserver(Observer):

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def observe_model(self, iteration, model):
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(self.X_test)
        print("Test Accuracy", accuracy_score(self.y_test, y_pred))

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug):
        pass
