def format_insert_query(table_name, values: dict):
    key_list = list(values.keys())

    query = f"INSERT INTO {table_name}"
    query = query + " (" + ", ".join(key_list) + ")"
    query = query + " VALUES ('" + "', '".join([str(values[key]) for key in key_list]) + "');"
    return query


def format_select_query(table_name, where: dict = None):
    query = f"SELECT * FROM {table_name}"
    if where is not None and len(where) > 0:
        query += " WHERE " + " AND ".join([key + "='" + str(value) + "'" for key, value in where.items()])
    query = query + ";"
    return query


def instantiate_class_by_fqn(learner_fqn, learner_params=None):
    import importlib

    module_name, class_name = learner_fqn.rsplit(".", 1)
    LearnerClass = getattr(importlib.import_module(module_name), class_name)
    instance = LearnerClass(**learner_params)
    return instance


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == "__builtin__":
        return klass.__name__
    return module + "." + klass.__name__
