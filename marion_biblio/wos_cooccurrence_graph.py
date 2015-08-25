
from cooccurrence import prune_o, inclusion_index_cooccurrence_matrix, cosine_cooccurrence_matrix, total_occurrences, cooccur_cutoff, cooccurrence_matrix, inverse_cooccurrence_matrix
from wos_reader import Wos_reader, cited_dois, authorlist_from_authorfield, download_names, Wos_h5_reader, wordbag, stems
from numpy import zeros, int32, int8, float32, array, sort
from itertools import chain
import pydot, re
import networkx as nx
from crossref import Crossref_browser

def cites_matrix( w ) :
    """
    Returns an occurrence matrix; the rows represent the papers in the corpus; the columns the cited papers.
    The entries are the intersections where a corpus paper cites a column paper
    """
    column_dois = w.set_cited_dois()
    row_dois = w.dois()
    result = zeros( (len(row_dois), len(column_dois)), dtype = int8 )
    for paper in w.reader() :
        if paper['DI'] != '' :
            for doi in cited_dois( paper ) :
                result[row_dois.index( paper['DI'] )][column_dois.index( doi )] = 1

    return ( result, row_dois, column_dois )

def sindices( a,b ) :
    """assumes a is sorted; returns indices (in a) of items in b which are also in a"""
    i = zip(b, a.searchsorted(b) )
    return [ x[1] for x in i if x[1] < a.shape[0] and x[0] == a[x[1]]]

def sindex(a,b) :
    """assumes a is sorted; returns index of b in a or None if not found"""
    i = a.searchsorted(b)
    return None if  i >= len(a) or a[i] != b else i


def w5_cites_matrix( w5 ) :
    rows = sort(w5.all_dois())
    cols = sort(w5.all_cited_dois())
    result = zeros((rows.shape[0], cols.shape[0]), dtype = int8)
    for paper in w5.h5.root.papers :
        rindex = sindex( rows, paper['doi'] )
        if rindex != None :
            for cindex in sindices( cols, w5.h5.root.cited_papers[paper['index']] ) :
                result[ rindex, cindex ] = 1
    return (result, rows, cols)

def internal_directed_cites_matrix( w ) :
    """
    returns a directed citations graph; this only involves papers listed in the data, and is a DIRECTED graph adgacency matrix
    such that cell (row,col) means "node row points to node col"
    """
    row_dois = w.dois()
    result = zeros( (len(row_dois), len(row_dois)), dtype=int8 )
    for paper in w.reader() :
        if paper['DI'] != '' :
            for doi in cited_dois(paper) :
                if row_dois.count( doi ) :
                    result[ row_dois.index(paper['DI']), row_dois.index(doi)] = 1
    return (result, row_dois, row_dois )

def w5_internal_directed_cites_matrix( w5 ) :
    rows = sort(w5.all_dois())
    cols = sort(w5.all_dois())
    result = zeros( ( rows.shape[0], cols.shape[0] ), dtype = int8 )
    for paper in w5.h5.root.papers :
        rindex = sindex( rows, paper['doi'] )
        if rindex != None :
            try :
                for cindex in sindices( cols, w5.h5.root.cited_papers[paper['index']] ) :
                    result[rindex,cindex] = 1
            except IndexError :
                print w5.h5.root.cited_papers[paper['index']]
    return (result, rows, cols )

def authors_by_doi( w ) :
    """returns a dictionary with keys of the DOIs and values of author lists"""
    Result = {}
    for paper in w.reader() :
        Result[ paper['DI'] ] = authorlist_from_authorfield( paper['AU'] )
    return Result

def internal_directed_authorcite_matrix( w ) :
    """
    returns a directed citations graph; this only involves authors
    such that cell (row,col) means "node row cites node col"
    """
    authordict = authors_by_doi( w )
    authors = list(w.set_authors())
    result = zeros( (len(authors), len(authors)), dtype=int32 )
    for paper in w.reader() :
        for node_author in authorlist_from_authorfield( paper['AU'] ) :
            for doi in cited_dois(paper) :
                if authordict.has_key(doi) :
                    for cited_author in authordict[doi] :
                        result[ authors.index( node_author ), authors.index( cited_author )] += 1
    return (result, authors, authors)

def w5_internal_directed_authorcite_matrix( w5 ) :
    rows = sort(w5.all_authors())
    cols = sort(w5.all_authors())
    authordict = w5.dict_doi_to_authors()
    result = zeros( (rows.shape[0], cols.shape[0]), dtype = int32 )
    for paper in w5.h5.root.papers :
        for rindex in sindices( rows, w5.h5.root.authors[paper['index']] ) :
            for doi in ( x for x in w5.h5.root.cited_papers[paper['index']] if authordict.has_key(x) ):
                for cindex in sindices( cols, authordict[doi] ) :
                    result[rindex,cindex] += 1
    return result, rows, cols


def pagerank_list( idcm, labels ) :
    """
    Takes an internal directed cite matrix and returns a sorted list of the rows by pagerank
    """
    g = nx.DiGraph( idcm )
    pr = nx.pagerank_numpy(g)
    l = list(pr.iteritems())
    # now l is a list of (index, pagerank)
    l.sort( lambda a,b : cmp( b[1],a[1] ) )
    return [ (x[0], labels[x[0]], x[1]) for x in l ]

def pagerank_list_with_doi_legend( l ) :
    result = [x + tuple([legend_line( x[0],x[1] )]) for x in l]
    return result

def authors_in_papers_matrix( w ) :
    """
    Returns an occurrence matrix; the rows represent the papers in the corpus; the columns the cited papers.
    The entries are the intersections where a corpus paper cites a column paper
    """
    authors = list(w.set_authors())
    row_dois = w.dois()
    result = zeros( (len(row_dois), len(authors)), dtype = int32 )
    for paper in w.reader() :
        if paper['AU'] != '' and paper['DI'] != '':
            for author in authorlist_from_authorfield( paper['AU'] ) :
                if author != '':
                    result[row_dois.index( paper['DI'] )][authors.index( author )] = 1

    return ( result, row_dois, authors )

def w5_authors_in_papers_matrix( w5 ) :
    rows = sort(w5.all_dois())
    cols = sort(w5.all_authors())
    result = zeros( (rows.shape[0], cols.shape[0]), dtype = int32 )
    for paper in w5.h5.root.papers :
        rindex = sindex( rows, paper['doi'] )
        if rindex != None :
            for cindex in sindices( cols, w5.h5.root.authors[paper['index']] ) :
                result[ rindex, cindex ] = 1
    return ( result, rows, cols )

def w5_title_stems_matrix( w5 ) :
    rows = sort( w5.all_dois() )
    cols = array(sort( w5.all_title_stems() ))
    result = zeros((rows.shape[0], cols.shape[0]), dtype= int32 )
    for paper in w5.h5.root.papers :
        rindex = sindex( rows, paper['doi'] )
        if rindex != None :
            for cindex in sindices( cols, array(stems(paper['title']))) :
                result[ rindex, cindex ] = 1
    return ( result, rows, cols )
            

def node_index( label ) :
    return int( re.match( "^(\d+)", label ).groups(0)[0] )

def legend_line( index, doi, browser=None ) :
    if (browser == None) :
        browser = Crossref_browser()
    browser.do_query(doi)
    if None in [browser.doi, browser.authors, browser.title, browser.journal, browser.date] :
        result = "Error on doi : "+doi
    else :
        result = str(index) + ")  " + browser.doi + ": " + browser.authors + ", " + browser.title + ".  " + browser.journal + ", " + browser.date
    return result

def doi_legend( nC, v, labels, max_edges = 10 ) :
    cutoff = cooccur_cutoff( nC, max_edges )
    coords = zip(*(nC >= cutoff).nonzero())
    dois = [labels[x] for x in set(chain.from_iterable(coords))]
    b = Crossref_browser()
    result = [legend_line( dois.index(doi) + 1, doi, b ) for doi in dois]
    return result

def networkx_cooccurrence_graph( nC, v, labels, max_edges = 10 ) :
    """
    Returns a Networkx-like undirected graph, to eventually be used for analysis and display
    """
    Result = nx.Graph()
    
    nv = v.astype( float32 ) / v.max()

    cutoff = cooccur_cutoff( nC, max_edges );

    coords = zip(*(nC >= cutoff).nonzero())

    # make a dict of all nodes which are mentioned in the coords
    nodes = {}
    index = 1
    # explicitly cast everything so that gexf and other files can convert correctly
    for coord in set(chain.from_iterable(coords)) :
        if not nodes.has_key( coord ) :
            Result.add_node( str(index), label=str(index), coord=str(coord), labelval=str(labels[coord]), index = str(index), width=float(nv[coord]) )
            nodes[ coord ] = str(index)
        index = index + 1

    for coord in coords :
        Result.add_edge( nodes[coord[0]], nodes[coord[1]], weight = float(nC[coord]), penwidth = float(nC[coord]) )

    return Result

def neato_from_networkx( g, min_node_size = 0.5, max_node_size = 2.0, min_edge_width = 1.0, max_edge_width = 5.0, legend_attribute=None, label_nodes_directly = False ) :
    d = nx.to_pydot( g )
    d.set_overlap(False)
    # normalize node size
    nodewidths = array( [float(n.get_width()) for n in d.get_nodes()] )
    # done this way in case input has all the same size to avoid divide by zero
    node_range = (nodewidths.max() - nodewidths.min())/(max_node_size - min_node_size)
    for n in d.get_nodes() :
        n.set_width( min_node_size + (float(n.get_width()) - nodewidths.min()) / node_range )
        n.set_fixedsize( "true" )
        n.set_shape('circle')
    # normalize edge width
    edge_widths = array( [float(e.get_penwidth()) for e in d.get_edges()] )
    edge_width_range = (edge_widths.max() - edge_widths.min())/(max_edge_width - min_edge_width)
    for e in d.get_edges() :
        e.set_penwidth( min_edge_width + (float(e.get_penwidth()) - edge_widths.min() )/edge_width_range )
    # if the legend attribute is set, create a legend node
    if label_nodes_directly :
        if legend_attribute == None :
            legend_attribute = 'labelval'
        for n in d.get_nodes() :
            n.set_label( n.get_attributes()[legend_attribute] )
    else : 
        legend = pydot.Node( "legend" )
        nodelist = [n.get_label()+": "+n.get_attributes()[legend_attribute] for n in d.get_nodes()]
        nodelist.sort( lambda a,b : cmp( int( a.split(':')[0] ), int (b.split(':')[0] ) ))
        
        legend.set_label(  "\l".join([x for x in nodelist])+"\l" )
        legend.set_shape("box")
        d.add_node(legend)
    return d
        

def neato_cooccurrence_graph( nC, v, labels, max_edges = 10, fnam_stem = "test", label_nodes_directly = False, scale=1.0, min_node_size = 0.1 ): 
    """
    makes a neato-style undirected graph from the given cooccurrence matrix, vector of 
    total occurrences,  and labels.  Assume C is normalized as desired and that
    all is pruned as desired!
    """
    
    nv = v.astype( float32 ) / v.max()

    cutoff = cooccur_cutoff( nC, max_edges );

    graph = pydot.Dot( graph_type = 'graph' )
    graph.set_overlap("false")
    coords = zip(*(nC >= cutoff).nonzero())

    # make a dict of all nodes which are mentioned in the coords
    nodes = {}
    index = 1
    for coord in set(chain.from_iterable(coords)) :
        if not nodes.has_key( coord ) :
            node =  pydot.Node( str(coord) )
            if v != None :
                #print coord
                label = labels[coord]
                if label_nodes_directly :
                    node.set_label( label )
                else :
                    node.set_label( str(index) )
                #node.set_penwidth( nv[ coord ] )
                node.set_fixedsize("true")
                node.set_width( max(min_node_size,scale *nv[ coord ]) )
                node.set_shape("circle")
            nodes[ coord ] = node
            graph.add_node( node )
        index = index + 1

    for coord in coords :
        
        edge = pydot.Edge( nodes[coord[0]], nodes[coord[1]] )
        edge.set_weight( nC[coord] )
        edge.set_penwidth( nC[coord]*5 )
        #edge.set_label( str(int(m[coord]) ))
        graph.add_edge(edge)

    if not label_nodes_directly : 
        legend = pydot.Node( "legend" )
        nodelist = nodes.items()
        nodelist.sort( lambda a,b : cmp(node_index(a[1].get_label()),node_index(b[1].get_label())) )
        legend.set_label(  "\l".join([x[1].get_label()+":"+labels[x[0]] for x in nodelist])+"\l" )
        legend.set_shape("box")
        graph.add_node(legend)

    #print graph.to_string()
    graph.write_dot(fnam_stem+'.dot', prog='neato' )
    graph.write_png(fnam_stem+'.png', prog='neato' )
    #graph.write_pdf(fnam_stem+'.pdf', prog='neato' )

import pystreamgraph
import random
import colorsys

def relabel_nx_graph( g, tag ) :
    """changes the "label" value for all nodes in a graph to the data of tag"""
    for item in g.node.iteritems() :
        item[1]['label'] = item[1][tag]

def country_streamgraph_data( pcc ) :
    data = []
    xs = pcc.keys()
    xs.sort( lambda a,b : cmp(int(a),int(b)) )
    xvals = [float(x) for x in xs]
    labels = pcc[xs[0]].keys()
    colors = []
    for layer in range(0,len(pcc.values()[0])) :
        ys = [pcc[x].values()[layer] for x in xs]
        data.append( zip(xvals,ys) )
        colors.append( colorsys.hsv_to_rgb(0.588,0.2, random.uniform(0.4,0.7)))
    return data, colors, labels
    

#w = Wos_reader(download_names("metamaterials",15))
#w = Wos_reader("metamaterials_cited.tab")
#w5 = Wos_h5_reader('metamaterials.h5')
#O,rows,cols = w5_authors_in_papers_matrix( w5 )
#O, rows, cols = w5_cites_matrix( w5 )
#O2, rows2, cols2 = prune_o(O,rows,cols,3)
#C = cooccurrence_matrix(O2)
#v2 = total_occurrences(O2)
#g = networkx_cooccurrence_graph( C, v2, cols2, 10000 )
#O,rows,cols = worked_with_matrix(w)
#O2, rows2, cols2 = prune_o( O, rows, cols, 3 )

#adict = authors_by_doi(w)
#O,rows,cols = internal_directed_authorcite_matrix(w)

#O, rows, cols = internal_directed_cites_matrix( w )
#pr = pagerank_list(O, rows)
#pr2 = pagerank_list_with_doi_legend( pr[0:5] )
#v = total_occurrences(O)

#O2, rows2, cols2 = prune_o( O, rows, cols, 10 )
#cC = cosine_cooccurrence_matrix( O2 )
#iC = inclusion_index_cooccurrence_matrix( O2 )
#C = cooccurrence_matrix( O2 )
#v2 = total_occurrences(O2)
#g = networkx_cooccurrence_graph( cC, v2, cols, max_edges = 10 ) 
#d = neato_from_networkx(g, max_edge_width = 20.0, label_nodes_directly = False, legend_attribute='labelval' )
#neato_cooccurrence_graph( iC, v, cols2, max_edges=1, fnam_stem="inclusion", label_nodes_directly = False, min_node_size=0.7, scale=3)
#d.write_png('test.png', prog='neato')
#legend = doi_legend( cC, v2, cols, max_edges = 10 )
