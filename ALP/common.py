def format_insert_query(table_name, values: dict):
    key_list = list(values.keys())

    query = f"INSERT INTO {table_name}"
    query = query + " (" + ", ".join(key_list) + ")"
    query = query + " VALUES ('" + "', '".join([str(values[key]) for key in key_list]) + "');"
    return query


def format_select_query(table_name, where: dict = None):
    query = f"SELECT * FROM {table_name}"
    if where is not None and len(where) > 0:
        query += " WHERE " + " AND ".join([key + "=' " + str(value) + "'" for key, value in where.items()])
    query = query + ";"
    return query
