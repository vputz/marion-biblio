"""A set of transforms and analysis methods to deal with cooccurrence analysis.

Cooccurrence analysis begins with a matrix O; the rows represent
documents and the columns represent items referenced by those
documents (cited documents, words in the abstract, etc).  The cells of
the occurrence matrix are typically filled with some value
representing the occurrences (normally "1")

Common terminology:
O represents the Occurrence matrix
C represents a Co-Occurrence matrix

"""

from numpy import array, triu, int32, float32, arange, newaxis, zeros, sqrt
import unittest
from itertools import chain


def prune_columns(O, row_labels, column_labels, cutoff=1):
    """Prune the occurrence matrix to eliminate columns ("total
    occurrences") less than a cutoff amount.  This is done to remove
    trivial entries and greatly reduce the size of the
    occurrence/cooccurrence matrices.  Trivial entries, particularly
    when normalized, can have disproportionately high cosine values
    (two papers cited in one paper and nowhere else) which is
    mathematically defensible but may not be useful when looking for
    relevant works.

    """
    v = O.sum(0)
    cols = (v > cutoff)

    O2 = O[:, cols]

    # now trim rows which no longer have entries
    v = O2.sum(1)
    rows = (v > 0)
    return O2[rows,:], list(array(row_labels)[rows]), list(array(column_labels)[cols])
    #return O[rows,cols], array(row_labels)[rows], array(column_labels)[cols]


def prune_rows(O, row_labels, column_labels, cutoff=1):
    """Prune the occurrence matrix to eliminate rows ("total
    occurrences") less than a cutoff amount.  This is done to remove
    trivial entries and greatly reduce the size of the
    occurrence/cooccurrence matrices.  Trivial entries, particularly
    when normalized, can have disproportionately high cosine values
    (two papers cited in one paper and nowhere else) which is
    mathematically defensible but may not be useful when looking for
    relevant works.

    """
    v = O.sum(1)
    rows = (v > cutoff)

    O2 = O[rows, :]

    # now trim columns which no longer have entries
    v = O2.sum(0)
    cols = (v > 0)

    return O2[:,cols], list(array(row_labels)[rows]), list(array(column_labels)[cols])
    #return O[rows,cols], array(row_labels)[rows], array(column_labels)[cols]


def num_gte( A, value ) :
    """
    returns the number of cells in array A greater than or equal to value 
    """
    return len( ( A >= value ).nonzero()[0] )

def cooccur_cutoff( nC, max_edges ) :
    """
    Attempts to find the cutoff which will result in about the max_nodes number of edges
    being selected in the graph.  Because equality comparisons are dicey with floats,
    this can be off a tiny bit but seems to work
    """
    # first get a sorted list of values
    l = list(set(chain.from_iterable( nC )))
    # if we could take all the edges, set the cutoff to be 1
    if ( max_edges >= len(l) ) :
        return 1
    l.sort(reverse = True)
    # the following commented code seems to work but is slow
    # now get the number of times each of these occurres
    #num_occur = lambda( x ) : len( (nC == x ).nonzero()[0] )
    #v = [ num_occur( x ) for x in l ]
    ## and the running sum of these
    #c = [ array(v[0:x]).sum() for x in range(1,len(l))]
    #result_index = array([x for x in takewhile( lambda x : c[x] < max_nodes, range(1,len(c)))]).max() 
    #return l[result_index]
    index = 0
    num_included = 0
    while num_included < max_edges :
        index = index + 1
        num_included = num_gte( nC, l[index] )
    return l[index-1]

def total_occurrences( O ) :
    """
    returns the total occurrences of each document in the occurrence matrix

    for citations, columns = occurrences of citation
    Returns a vector of the total citations of each paper in the matrix (summing on columns).
    For each, this should be equal to or less than Z9 (Z9 reflects total cites, and this
    only reflects cites in this corpus)
    """
    return O.sum(0)

def cooccurrence_matrix( O ) :
    """
    Returns a cooccurrence matrix.

    Given the occurrence matrix O, this is just ( OT * O ), and we make it upper triangular
    to simplify things
    """
    # theano:
    # o = T.matrix('o')
    # result = T.triu( o.T.dot(o), 1 )
    # f = function( [o], result )
    return triu( O.T.dot(O),1 )


def inverse_cooccurrence_matrix( O ) :
    """
    Returns an "inverse" cooccurrence matrix.  In other words, if the rows of O represent papers and
    columns the papers they cite, then the usual coccurrence matrix is "papers which are cited together"
    and is useful for getting a background in a subject.
    
    The "inverse" cooccurrence matrix is "papers which cite the same sorts of papers" and may be more
    useful in gathering information about newer papers.

    This doesn't seem to be very useful.
    """
    return triu( O.dot(O.T), 1 )

def cosine_cooccurrence_matrix( O ) :
    """
    Returns a cooccurrence matrix normalized via cosine

    S_ij = c_ij / sqrt(s_i*s_j)

    """
    C = cooccurrence_matrix(O)
    v = sqrt(total_occurrences(O))

    result = ( C.astype(float32) / v) / v[:,newaxis]
    return result

def association_index_cooccurrence_matrix(O):
    """
    Returns a cooccurrence matrix normalized via association index (see van Eck)
    S_ij = c_ij / si*sj
    Arguments:
    - `O`: matrix
    """
    C = cooccurrence_matrix( O )
    v = total_occurrences(O)
    result = ( C.astype(float32) / v ) / v[:,newaxis]
    return result
    

def minarray(v) :
    """
    Passed an N-length vector v, eturns a NxN array C
    where Sij = c_ij / min(v_i,v_j)
    """
    result = zeros( (len(v), len(v) ) )
    for i in arange( len(v) ) :
        for j in arange( len(v) ) :
            result[i,j] = min(v[i],v[j])

    return result

def inclusion_index_cooccurrence_matrix( O ) :
    """
    Returns a cooccurrence matrix normalized via the inclusion index
    """
    C = cooccurrence_matrix(O)
    v = total_occurrences(O)
    minv = minarray(v)
    return C.astype(float32)/minv

class TestCooccurrence( unittest.TestCase ) :

    def setUp(self) :
        self.O = array( [[1,1,0,1],[1,1,1,0]], dtype=int32 )
        self.row_labels = ['A','B']
        self.column_labels = ['a','b','c','d']

    def test_total_occurrences( self ) :
        v = total_occurrences(self.O) 
        self.assertTrue( (v == array([2,2,1,1]) ).all() )

    def test_inclusion_index_cooccurrence_matrix( self ) :
        C = inclusion_index_cooccurrence_matrix( self.O )
        test = array([[0,1,1,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]])
        self.assertTrue( (C==test).all() )

    def test_cooccurrence_matrix( self ) :
        C = cooccurrence_matrix( self.O )
        test = array([ [0, 2, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0,0,0,0]] )
        self.assertTrue( (C==test).all() )

    def test_cosine_cooccurrence_matrix( self ) :
        nC = cosine_cooccurrence_matrix( self.O )
        test = array([ [0,0.5,0.5,0.5],[0,0,0.5,0.5],[0,0,0,0],[0,0,0,0]] )

    def test_prune_columns( self ) :
        pO, p_rlabels, p_clabels = prune_columns( self.O, self.row_labels, self.column_labels )
        self.assertTrue( (pO == array([[1,1],[1,1]])).all() )
        self.assertEqual( p_clabels, ['a','b'] )

        o,rlabels,clabels = prune_columns( array([[1,1,0,0],[0,0,0,1],[1,1,1,0]]), 
                                     ['A','B','C'],
                                     ['a','b','c','d'] )
        self.assertTrue( (o==array([[1,1],[1,1]])).all() )
        self.assertEqual( clabels, ['a','b'] )
        self.assertEqual( rlabels, ['A','C'] )

    #def test_node_index( self ) :
    #    self.assertEqual( node_index("13:blorp"), 13  )

#if __name__ == "__main__" :
#    unittest.main()
