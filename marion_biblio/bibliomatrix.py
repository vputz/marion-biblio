from marion_biblio.cooccurrence import prune_rows, prune_columns, \
    cooccurrence_matrix, cosine_cooccurrence_matrix,\
    association_index_cooccurrence_matrix, inclusion_index_cooccurrence_matrix
from pyrsistent import pvector
from enum import Enum
import numpy
from abc import ABCMeta, abstractmethod
import networkx as nx
import igraph


class BiblioMatrix:
    __metaclass__ = ABCMeta
    """
    An augmented matrix class which stores the matrix itself and then
    identifiers (strings) for rows and columns, so that it is possible
    to see what the
    rows and columns refer to.
    """

    def __init__(self, mat, rows, cols):
        target_type = {numpy.dtype('int8'): numpy.dtype('int64'),
                       numpy.dtype('int16'): numpy.dtype('int64'),
                       numpy.dtype('int32'): numpy.dtype('int64'),
                       numpy.dtype('int64'): numpy.dtype('int64'),
                       numpy.dtype('float16'): numpy.dtype('float32'),
                       numpy.dtype('float32'): numpy.dtype('float32'),
                       numpy.dtype('float64'): numpy.dtype('float64')}
        self._matrix = mat.astype(target_type[mat.dtype])
        self._rows = pvector(rows)
        self._columns = pvector(cols)

    @property
    def matrix(self):
        return self._matrix

    @property
    def rows(self):
        return self._rows

    @property
    def columns(self):
        return self._columns

    @property
    def is_consistent(self):
        return (len(self._rows) == self._matrix.shape[0]) \
            and (len(self._columns) == self._matrix.shape[1])

    def __str__(self):
        return "\n".join(["{0}x{1} {2}:".format(self.matrix.shape[0],
                                                self.matrix.shape[1],
                                                self.__class__.__name__),
                          repr(self.matrix),
                          "Rows:",
                          repr(self.rows),
                          "Cols:",
                          repr(self.columns)])

    def row_pruned(self, cutoff=1):
        o, rows, cols = prune_rows(self.matrix, self.rows, self.columns,
                                   cutoff=cutoff)
        return type(self)(o, rows, cols)

    def column_pruned(self, cutoff=1):
        o, rows, cols = prune_columns(self.matrix, self.rows, self.columns,
                                      cutoff=cutoff)
        return type(self)(o, rows, cols)

    def column_pruned_to_top_fraction_of_occurrences(self, fraction=0.1, max_columns=20):
        v = numpy.sort(self.matrix.sum(0))
        v = v[::-1]
        total = v.sum()
        target_occurrences = total*fraction
    
        acc = 0
        index = 0
        while acc < target_occurrences:
            acc = acc + v[index]
            index=index+1
        index = min(index, max_columns)
        # now prune everything with occurrences less than v[index]
        return self.column_pruned(v[index])

    def zeroed_if_less_than(self, value):
        m = self.matrix.copy()
        m[m < value] = 0
        return type(self)(m, self.columns, self.rows)

    def transposed(self):
        return type(self)(self.matrix.T, self.columns, self.rows)


class CooccurrenceType(Enum):
    simple = 1,
    cosine_index = 2,
    association_index = 3,
    inclusion_index = 4


class OccurrenceMatrix(BiblioMatrix):
    """
    A class meant to encode occurrences, ie "X occurs in Y"; the X's
    are the columns, and the Y's are the rows (documents); in other words,
    cases such as

    words (cols) occurring in paper abstracts (rows)
    authors (cols) working on papers (rows)
    papers (cols) cited by other papers (rows)

    USUALLY an occurrence matrix will have binary entries; "X occurs in Y";
    it can be useful to make inclusion matrices ("X occurs in Y Z times") but
    the analysis of that is not always obvious
    """

    def __init__(self, mat, rows, cols):
        super().__init__(mat, rows, cols)

    def column_cooccurrence(self, cooccurrence_type=CooccurrenceType.simple):
        lookup = {CooccurrenceType.simple: cooccurrence_matrix,
                  CooccurrenceType.cosine_index: cosine_cooccurrence_matrix,
                  CooccurrenceType.association_index: association_index_cooccurrence_matrix,
                  CooccurrenceType.inclusion_index: inclusion_index_cooccurrence_matrix}
        o = lookup[cooccurrence_type](self.matrix)
        return CooccurrenceMatrix(o + o.T, self.columns,
                                  self.columns, cooccurrence_type)



class CooccurrenceMatrix(BiblioMatrix):
    """Returns a "cooccurrence" matrix; Occurrence matrices are bipartite
    graphs, which are in general unweighted (could be weighted),
    asymmetric, and directed ("x occurs in y") while cooccurrence
    matrices are weighted, symmetric, and undirected ("x and y occur
    together N times")

    """

    def __init__(self, mat, rows, cols, cooccurrence_type):
        super().__init__(mat, rows, cols)
        self._cooccurrence_type = cooccurrence_type

    @property
    def is_symmetric(self):
        return numpy.allclose(self.matrix.transpose(), self.matrix) \
            and self.rows == self.columns

    @property
    def is_consistent(self):
        return super().is_consistent and \
            self.is_symmetric and \
            self.rows == self.columns

    def __str__(self):
        return super().__str__() + "\n" + "Type: " + \
            repr(self.cooccurrence_type)

    @property
    def cooccurrence_type(self):
        return self._cooccurrence_type

    def as_nx_graph(self):
        result = nx.from_numpy_matrix(self.matrix)
        relabel_dict = dict(zip(numpy.arange(len(result.nodes())),
                                self.rows))
        nx.relabel_nodes(result,
                         dict(zip(
                             numpy.arange(len(result.nodes())),
                             self.rows)),
                         False) # relabel in place
        return result

    def as_igraph(self):
        result = igraph.Graph.Weighted_Adjacency(self.matrix.tolist(), mode=igraph.ADJ_MAX)
        result.vs["name"]=self.rows
        return result
