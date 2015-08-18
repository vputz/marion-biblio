

def pv_remove(pv, *items):
    result = pv
    for item in items:
        result = result.delete(result.index(item))
    return result
