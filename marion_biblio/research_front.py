from numpy import sort, zeros, int8
from marion_biblio.bibliomatrix import OccurrenceMatrix


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


def w5_cites_matrix(w5):
    """Creates a citation matrix from the corpus, where columns represent
    cited papers and rows represent papers doing the citing

    """
    rows = sort(w5.all_dois())
    cols = sort(w5.all_cited_dois())
    result = zeros((rows.shape[0], cols.shape[0]), dtype=int8)
    for paper in w5.h5.root.papers:
        rindex = sindex(rows, paper['doi'])
        if rindex is not None:
            for cindex in sindices(
                    cols, w5.h5.root.cited_papers[paper['index']]):
                result[rindex, cindex] = 1
    return OccurrenceMatrix(result, rows, cols)
