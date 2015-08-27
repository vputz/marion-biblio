def sindices(a, b):
    """assumes a is sorted; returns indices (in a) of items in b which are
    also in a

    """
    i = zip(b, a.searchsorted(b))
    return [x[1] for x in i if x[1] < a.shape[0] and x[0] == a[x[1]]]


def sindex(a, b):
    """assumes a is sorted; returns index of b in a or None if not found"""
    i = a.searchsorted(b)
    return None if i >= len(a) or a[i] != b else i


def pv_remove(pv, *items):
    result = pv
    for item in items:
        result = result.delete(result.index(item))
    return result
