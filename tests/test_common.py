from sklearn.ensemble import RandomForestClassifier

from alpbench.util.common import format_insert_query, format_select_query, fullname, instantiate_class_by_fqn


def test_insert_query():
    query = format_insert_query("TBL", {"x": 1})
    assert query == "INSERT INTO TBL (x) VALUES ('1');"


def test_select_query():
    query = format_select_query("TBL", where={"x": 1})
    assert query == "SELECT * FROM TBL WHERE x='1';"

    query = format_select_query("TBL")
    assert query == "SELECT * FROM TBL;"


def test_instantiate_class_by_fqn():
    rf = instantiate_class_by_fqn("sklearn.ensemble.RandomForestClassifier")
    assert type(rf) is RandomForestClassifier


def test_fullname_of_builtin():
    assert fullname("a") == "builtins.str"
