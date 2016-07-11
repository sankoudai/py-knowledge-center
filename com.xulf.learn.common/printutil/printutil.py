__author__ = 'quiet road'


def printVar(v, fmt=None):
    pat_string = "type: {}    value: {}"
    if fmt:
        pat_string = "type: {}    value: {:" + fmt + "}"

    var_string = pat_string.format(type(v), v)

    print(var_string)
