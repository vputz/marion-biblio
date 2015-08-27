from numpy import sort, zeros, int8
from marion_biblio.bibliomatrix import OccurrenceMatrix, CooccurrenceType
from marion_biblio import crossref
from marion_biblio.helpers import sindices, sindex
# one reference: http://archive.sciencewatch.com/about/met/rf-methodology/


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


def entry_from_vertex(v):
    info = crossref.doi_info_or_none(v["name"])
    vresult = dict(doi=v["name"].decode('utf-8'),
                   title=crossref.title_from_info(info),
                   author=crossref.authors_from_info(info))
    if "weight" in v.attributes():
        vresult['weight'] = int(v['weight'])
    return vresult


def coc_to_research_front(coc, vertex_weights=None):
    g = coc.as_igraph()
    if vertex_weights is not None:
        g.vs["weight"] = vertex_weights
    comm = g.community_walktrap(weights="weight").as_clustering()
    result = []

    for s in comm.subgraphs():
        this_group = [entry_from_vertex(v) for v in s.vs]
        if vertex_weights is not None:
            this_group.sort(key=lambda x: x['weight'], reverse=True)
        result.append(this_group)

    return result


def basic_research_front(oc, fraction=0.1, max_columns=20,
                         index_type=CooccurrenceType.cosine_index):
    """Take an occurrence matrix and return a research front using the top
    "fraction" of occurrences and the max number of columns; this is a
    pragmatic measure to reduce the number of output entries

    """
    oc2 = oc.column_pruned_to_top_fraction_of_occurrences(fraction,
                                                          max_columns)
    coc = oc2.column_cooccurrence(index_type)
    return coc_to_research_front(coc, vertex_weights=oc2.matrix.sum(0))


def basic_research_front_from_w5(w5):
    oc = w5_cites_matrix(w5)
    return basic_research_front(oc, fraction=0.15)
