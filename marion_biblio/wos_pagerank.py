import numpy
from marion_biblio.research_front import sindex, sindices
import logging
from marion_biblio import bibliomatrix
from marion_biblio import crossref

def w5_internal_directed_cites_matrix(w5):
    """Returns a square matrix where the rows and columns are all DOIs.
    Each cell represents a case where a row paper cites a column
    paper, so this is a directed graph.

    """

    rows = numpy.sort(w5.all_dois())
    cols = rows.copy()
    result = numpy.zeros((rows.shape[0], cols.shape[0]), dtype=numpy.int8)
    for paper in w5.h5.root.papers:
        rindex = sindex(rows, paper['doi'])
        if rindex is not None:
            try:
                for cindex in sindices(
                        cols, w5.h5.root.cited_papers[paper['index']]):
                    result[rindex, cindex] = 1
            except IndexError:
                logging.error(w5.h5.root.cited_papers[paper['index']])
    return (result, rows, cols)


def w5_internal_directed_authorcite_matrix(w5):
    """Returns a square matrix where the rows and columns are all author
    names; each cell represents a case where a row author cites a
    column author, so this is a directed graph.  Unlike the DOI case,
    though, here authors can cite each other multiple times

    """

    rows = numpy.sort(w5.all_authors())
    cols = rows.copy()
    authordict = w5.dict_doi_to_authors()
    result = numpy.zeros((rows.shape[0], cols.shape[0]),
                         dtype=numpy.int32)
    for paper in w5.h5.root.papers:
        for rindex in sindices(rows, w5.h5.root.authors[paper['index']]):
            for doi in (x for x in w5.h5.root.cited_papers[paper['index']]
                        if x in authordict):
                for cindex in sindices(cols, authordict[doi]):
                    result[rindex, cindex] += 1
    return result, rows, cols


def papers_pagerank_query(w5, max_papers=20):
    m, r, c = w5_internal_directed_cites_matrix(w5)
    cm = bibliomatrix.CitationMatrix(m, r, c)
    pr = cm.pagerank()

    def entry_from_doi(doi):
        info = crossref.doi_info_or_none(doi)
        try:
            vresult = dict(doi=doi.decode('utf-8'),
                           title=crossref.title_from_info(info),
                           author=crossref.authors_from_info(info))
            return vresult
        except:
            logging.exception("Error on info {0}".format(info))
            return None

    return [(x[0], entry_from_doi(x[1]))
            for x in pr[0:min(max_papers, len(pr))]]


def authors_pagerank_query(w5, max_authors=20):
    m, r, c = w5_internal_directed_authorcite_matrix(w5)
    cm = bibliomatrix.CitationMatrix(m, r, c)
    pr = cm.pagerank()

    return pr[0:min(max_authors, len(pr))]
