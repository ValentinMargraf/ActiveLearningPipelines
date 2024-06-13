def format_insert_query(table_name, values: dict):
    """
    Format an insert query for a given table and values.

    Parameters:
        table_name (str): The name of the table to insert into.
        values (dict): A dictionary of column names and values to insert.

    Returns:
        str: The formatted insert query.
    """
    key_list = list(values.keys())

    query = f"INSERT INTO {table_name}"
    query = query + " (" + ", ".join(key_list) + ")"
    query = query + " VALUES ('" + "', '".join([str(values[key]) for key in key_list]) + "');"
    return query


def format_select_query(table_name, where: dict = None):
    """
    Format a select query for a given table and where clause.

    Parameters:
        table_name (str): The name of the table to select from.
        where (dict): A dictionary of column names and values to filter by.

    Returns:
        str: The formatted select query.
    """
    query = f"SELECT * FROM {table_name}"
    if where is not None and len(where) > 0:
        query += " WHERE " + " AND ".join([key + "='" + str(value) + "'" for key, value in where.items()])
    query = query + ";"
    return query


def instantiate_class_by_fqn(learner_fqn, learner_params={}):
    """
    Instantiate a class by its fully qualified name.

    Parameters:
        learner_fqn (str): The fully qualified name of the class to instantiate.
        learner_params (dict): A dictionary of parameters to pass to the class constructor

    Returns:
        object: The instantiated class.
    """
    import importlib
    module_name, class_name = learner_fqn.rsplit(".", 1)
    LearnerClass = getattr(importlib.import_module(module_name), class_name)
    instance = LearnerClass(**learner_params)
    return instance


def fullname(o):
    """
    Get the fully qualified name of a class.

    Parameters:
        o (object): The object to get the fully qualified name of.

    Returns:
        str: The fully qualified name of the class.
    """
    klass = o.__class__
    module = klass.__module__
    if module == "__builtin__":
        return klass.__name__
    return module + "." + klass.__name__
