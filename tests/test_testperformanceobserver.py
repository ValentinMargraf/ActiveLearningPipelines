from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from alpbench.benchmark.Observer import PrintObserver


def test_performance_evaluation():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    performance_observer = PrintObserver(X_test=X_test, y_test=y_test)
    performance_observer.observe_model(model=model, iteration=0)
