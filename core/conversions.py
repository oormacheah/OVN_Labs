from numpy import log10


def lin2db(linear_value):
    return 10 * log10(linear_value)


def db2lin(db_value):
    return 10 ** (db_value / 10)

