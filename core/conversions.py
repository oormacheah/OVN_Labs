from numpy import log10


def lin2db(linear_value):
    return 10 * log10(linear_value)


def db2lin(db_value):
    return 10 ** (db_value / 10)


def path_str2arrow(in_path):
    path_arrow_str = ''
    for node in in_path:
        path_arrow_str += node + '->'
    return path_arrow_str[:-2]


def path_arrow2str(in_path):
    return in_path[::3]
