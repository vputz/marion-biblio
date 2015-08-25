import sys
sys.path.append("/home/vputz/geocash")

import unittest

import csv, re
from collections import Counter
from itertools import chain, combinations, takewhile
from matplotlib import pyplot as plt
from numpy import arange, array, zeros, sqrt, triu, float32, int32, newaxis, sqrt, take, asarray, sum, log
from scipy.linalg import svd
from geocash import get_latlong, read_remap_data, write_remap_data
from googlemaps import GoogleMapsError
from mpl_toolkits.basemap import Basemap
import pydot

from matplotlib import pyplot as plt
"""
A key to the WoK fields, http://images.webofknowledge.com/WOKRS57B4/help/WOS/hs_wos_fieldtags.html
FN File name
VR Version Number
PT Publication Type (J = journal B = book, S = Series, P=Patent)
AU Authors last name/initial
AF Authors, full name
BA Book Authors
BF Book authors, full name
CA Group Authors
GP Book Group Authors
BE Editors
TI title
SO Source (journal)
SE Series title
LA language
DT Document Type
CT Conferency Title
CY Conference Date
CL Conference Location
SP Conference Sponsors
HO Conference Host
DE Author Keywords
ID KeyWords Plus
AB abstract
C1 Addresses (of authors)
RP Reprint Address
EM Email contact
FU Funding and grant number
FX Funding text
CR Cited References
NR Cited Reference Count
TC WOS Times CIted count
Z9 Total times cited count (WoS, BCI, CSCD)
PU Publisher
PI Publisher City
PA Publisher Address
SN International Standard Serial Number (ISSN)
BN International Standard Book Number (ISBN)
J9 29-char source abbreviation
JI ISO source abbreviation
PD Publication date
PY Year Published
VL Volume
IS Issue
SI Special issue
PN Part Number
SU Supplement
MA Meeting Abstract
BP Beginning Page
EP Ending Page
AR Article Number
DI Digital Object Identifier (DOI)
D2 Book digital object identifier (DOI)
PG Page count
P2 Chapter Count
WC Web of Science categories
SC research areas
GA document delivery number
UT accession number
ER end of record
EF end of file
"""


def import_wos( fnam ) :
    f = open(fnam, 'rb')
    r = csv.DictReader(f, delimiter = "\t" )
    result = [x for x in r]
    sort_data_by_citations(result)
    return result

def cmp_numcite( a, b ) :
    """Descending comparison in number of citations"""
    return cmp( int(b['Z9']), int(a['Z9']) )

def sort_data_by_citations( data ) :
    """Sorts high->Low"""
    return data.sort( cmp_numcite )
    
def pubs_by_articles_published( data ) :
    """Returns a list of publications with articles published, sorted in descending order
    (ie top entry has the most publications)"""
    # let's be Pythonic and use counter
    result = [ (k,v) for k,v in Counter([x['SO'] for x in data]).iteritems() ]
    # now sort
    result.sort( lambda a,b : cmp(b[1],a[1]) )
    return result

def authors(item) :
    return item['AU'].lower().split('; ')

def all_authors( data ) :
    """Returns a list of all authors in the corpus of data"""
    return list(set( chain.from_iterable( [ authors(x) for x in data ] ) ))

def author_in( author, item ) :
    return author.upper() in item['AU'].upper().split('; ')

def papers_and_cites( author, data ) :
    return [(x['DI'], x['Z9']) for x in data if author_in( author, x ) ]

def local_h( author, data ) :
    """returns a tuple; the first is the "Local H index" which for a small sample is
    likely just the number of papers; the second is the SMALLEST number of cites,
    so someone with three papers the least of which was cited 100 times will be
    ranked higher than someone with three papers the least of which was cited once"""
    papers = papers_and_cites( author, data )
    papers.sort( lambda a,b : cmp(int(b[1]), int(a[1])) )
    i = 0
    while i < (len(papers)-1) and i < int(papers[i][1] ) :
        i = i + 1

    return (i+1, int(papers[i][1]) )

def cmp_h( a, b ) :
    result = cmp( a[0], b[0] )
    return cmp( a[1], b[1] ) if result == 0 else result

def authors_by_h( data ) :
    authors = all_authors(data)

    result = [ (a,local_h(a,data)) for a in authors ]
    result.sort( lambda a, b : cmp_h(b[1], a[1]) )
    return result

def primary_author( item, data ) :
    authors = item['AU'].upper().split('; ')
    result = [ (a,local_h(a,data)) for a in authors ]
    result.sort( lambda a, b : cmp_h(b[1], a[1]) )
    return result[0][0]

def first_author( item ) :
    authors = item['AU'].upper().split('; ')
    return authors[0]

def print_top_pubs_table( data ) :
    row_format = "{:<20}{:<70}{:<6}{:<6}"
    print row_format.format( "Author", "Title", "Year", "Cites" )
    for item in data[0:10] :
        print row_format.format( primary_author(item,data), item['TI'], item['PY'], item['Z9'] )

def addresses( data ) :
    """retrieves a flat list of addresses from the data; this is taken from 
    splitting the list at 'C1' and then removing names and leading/trailing
    whitespace"""
    return list( set(chain.from_iterable( [ re.sub(r'\[.*?\]\s+','',x['C1']).split('; ') for x in data ] )))

def address_occurrences( address, data ) :
    return len( [x for x in data if x['C1'].upper().find(address.upper()) > -1] )

def addresses_with_numbers( data ) :
    a = [x for x in addresses(data) if x != '']
    numbers = [address_occurrences(b,data) for b in a]
    result = zip(a,numbers)
    result.sort( lambda a,b : cmp(b[1],a[1]) )
    return result
    
def addresses_with_locations( data, cache_fnam = "lookup_cache.csv" ) :
    a = addresses_with_numbers( data )
    cache = read_remap_data( cache_fnam )
    result = []
    for entry in a :
        try :
            location = get_latlong( entry[0], cache=cache )
            result.append( (entry[0], entry[1], location) )
        except GoogleMapsError as e2 :
            print e2
            print "Failed to Locate ", entry[0]
    write_remap_data( cache, cache_fnam )
    return result

def address_map( data ) :
    a = addresses_with_locations( data )
    plt.figure()
    m = Basemap( projection = "robin",
                 llcrnrlon = -180,
                 llcrnrlat = -80,
                 urcrnrlon = 180,
                 urcrnrlat = 80,
                 lon_0 = 0,
                 lat_0 = 0,
                 resolution = "c" )

    m.drawcoastlines()
    m.fillcontinents( color = 'coral', lake_color = 'aqua', alpha=0.3 )
    m.drawmeridians( arange( -180, 180, 60 ) )
    m.drawparallels( arange( -80, 80, 60 ) )

    xys = array([m(x[2][1], x[2][0]) for x in a])
    xs = xys[:,0]
    ys = xys[:,1]
    counts = array([x[1] for x in a]) * 200

    m.scatter( xs, ys, s=counts, marker='o', c='b', alpha=0.3 )
    plt.title("Research by sites, top 500 papers Ultraintense Lasers")
    plt.savefig("map.png")
    plt.close()

def top_pubs_graph( data, filename="test.png", subject = "Ultraintense Lasers", num_bars = 10 ) :
    d = pubs_by_articles_published( data )[0:num_bars]
    d.reverse()
    ypos = arange( num_bars ) + 0.5
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust( left=0.315 )
    rects = ax.barh( ypos, [x[1] for x in d], align='center', height=0.5, color='b' )
    ax.set_title("Top pubs by cites: " + subject )
    ax.set_yticks( ypos )
    ax.set_yticklabels( [x[0] for x in d], fontsize = "small" )

    # now plot the numbers at the right side
    for rect in rects : 
        width = int(rect.get_width())
        if (width < 5) :
            xloc = width + 1
            clr = 'black'
            align = 'left'
        else :
            xloc = 0.98*width
            clr = 'white'
            align = 'right'
        yloc = rect.get_y() + rect.get_height() /2.0
        ax.text( xloc, yloc, str(width), horizontalalignment=align, 
                 verticalalignment='center', color=clr, weight='bold' )

    fig.savefig(filename)

def quick_pubs_graph( subject ) :
    top_pubs_graph( import_wos( subject+".tab" ), subject+"_pubs.png", subject=subject, num_bars=8 )

def cited_dois( item ) :
    return re.findall('DOI\s*([^\s;]*)',item['CR'])

def all_cited_dois( corpus ) :
    """Gets a lits of all cited papers of all papers in a data set"""
    cites = list(set(chain.from_iterable((cited_dois(x) for x in corpus))))
    return cites

def cites_matrix( w ) :
    """
    Returns an occurrence matrix; the rows represent the papers in the corpus; the columns the cited papers.
    The entries are the intersections where a corpus paper cites a column paper
    """
    column_dois = w.set_cited_dois()
    row_dois = w.dois()
    result = zeros( (len(row_dois), len(column_dois)), dtype = int32 )
    for paper in w.reader() :
        if paper['DI'] != '' :
            for doi in cited_dois( paper ) :
                result[row_dois.index( paper['DI'] )][column_dois.index( doi )] = 1

    return ( result, row_dois, column_dois )

my_ignore_words = set(('a','von','','the','and','from','with','to','some','for','of','by','on','in','all','do','we','is','it','if','has','was','no','can','so','not','one','any','or','an','as','at','are','us','our','elsevier','institute' ,'edition','little'))

def stripword( s ) :
    """
    strips punctuation from word
    """
    return re.sub( '[\W\d]', '', s )

def wordbag( text, ignore_words = Ignore_words ) :
    """
    A dictionary where the keys are words and the values are counts of words in the text.
    Taking the keys() should get a list of unique words
    """
    iter = (stripword(s) for s in text.lower().split() if stripword(s) not in ignore_words)
    result = {}
    for x in iter :
        if result.has_key(x) :
            result[x] += 1
        else :
            result[x] = 1
    return result

Ignore_words = wordbag( open('english_stop.txt').read()+" ".join(my_ignore_words), () ).keys()

def words( item, key, ignore_words = Ignore_words ) :
    result = list(wordbag(item[key], ignore_words).keys())
    result.sort()
    return result

def all_words( corpus, key, ignore_words = Ignore_words ) :
    """
    Set of all words in all items referenced by key
    """
    return list(set(chain.from_iterable( (words(c,key,ignore_words) for c in corpus ) ) ) )
    
def words_matrix( corpus, key, cooccurrence_only = True, ignore_words = Ignore_words ) :
    """
    Returns an occurrence matrix O with the rows representing papers and the columns
    the words in the titles.  ignore_words is a list of words to ignore
    """
    atw = all_words(corpus, key, ignore_words)
    atw.sort()
    row_dois = [x['DI'] for x in corpus]
    result = zeros( (len(corpus),len(atw)), dtype = int32 )
    for paper in corpus :
        for word, occurrences in wordbag( paper[key], ignore_words ).iteritems() :
            result[ row_dois.index( paper['DI'] ) ][ atw.index( word ) ] = occurrences

    if cooccurrence_only :
        result[ result > 1 ] = 1
    return result, row_dois, atw

def title_words_matrix( corpus, cooccurrence_only = True, ignore_words = Ignore_words ) :
    """
    Returns an occurrence matrix O with the rows representing papers and the columns
    the words in the titles.  ignore_words is a list of words to ignore
    """
    return words_matrix( corpus, 'TI', cooccurrence_only, ignore_words )

def abstract_words_matrix( corpus, cooccurrence_only = True, ignore_words = Ignore_words ) :
    """
    Returns an occurrence matrix O with the rows representing papers and the columns
    the words in the abstracts.  ignore_words is a list of words to ignore
    """
    return words_matrix( corpus, 'AB', cooccurrence_only, ignore_words )

def authors_matrix( corpus ) :
    """
    Returns an occurrence matrix O with the rows representing papers
    and the columns the authors of those papers.
    """
    all = all_authors(corpus)
    row_dois = [x['DI'] for x in corpus]
    result = zeros( (len(corpus),len(all)), dtype = int32 )
    for paper in corpus :
        for item in authors( paper ) :
            result[ row_dois.index( paper['DI'] ) ][ all.index( item ) ] = 1

    return result, row_dois, all
    


def tfidf_occurrence_matrix( O ) :
    """
    Returns a cooccurrence matrix normalized via TFIDF
    TFIDFi,j = (Ni,j / N*j)*log(D/Di)
      Ni,j = number of times word i appears in document j
      N*j = number of total words in document j
      D = number of documents
      Di = number of documents in which word i appears (nonzero rows in column i)
    """
    # number of words in each document
    words_in_doc = O.sum(1)
    docs_containing_word = sum( asarray( O > 0, 'i' ), axis=0 )
    logpart = log(float(O.shape[0]) / docs_containing_word )

    result = (O.astype(float32) / words_in_doc[:,newaxis] ) * logpart
    
    return result
    
    
    

def normalized_cooccurrence_matrix( O, prune=3 ) : 
    """
    normalizes the cocitation matrix by the association strength 
      - Sa(c_ij, s_i, s_j) = c_ij / (s_i * s_j) : Association strength - Probabilistic similarity measure

      actually for a while I'm going with the cosine, since we want some measure of paper popularity and not 
      just cocite

    cocite_m is an upper triangular cocite matrix, cites_v is a vector of cites
    """
    C = cooccurrence_matrix( O )
    # get rid of singles
    C[C<=prune]=0
    v = total_occurrences( O )
    sv = sqrt(v)
    assert( len(C.shape) == 2 )
    assert( C.shape[0] == len(v) )
    assert( C.shape[1] == len(v) )
    
    result =  ( C.astype( float32 ) / sv ) / sv[:,newaxis] 
    # now renormalize to 1
    return result / result.max()

    #result = zeros( cocite_m.shape, dtype=float32 )
    #for row in arange( cocite_m.shape[1] ) :
    #    for col in arange( cocite_m.shape[0] ) :
    #        result[row,col] = float(cocite_m[row,col]) / (cites_v[row]*cites_v[col])

    #return result

def nodes_from_c( C ) :
    return len(set(chain.from_iterable( zip(*C.nonzero()) )))

def short_title( doi, corpus ) :
    items = [x for x in corpus if x['DI']==doi]
    if len(items) == 0 : 
        return ""
    else :
        return first_author(items[0]) + " " + items[0]['PY']

        

def neato_cooccurrence_graph( nC, v, labels, max_nodes = 10, fnam_stem = "test", label_nodes_directly = False, scale=1.0, min_node_size = 0.1 ): 
    """
    makes a neato-style undirected graph from the given cooccurrence matrix, vector of 
    total occurrences,  and labels.  Assume C is normalized as desired and that
    all is pruned as desired!
    """
    
    nv = v.astype( float32 ) / v.max()

    cutoff = cooccur_cutoff( nC, max_nodes );

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
    

def neato_graph_from_corpus( corpus, max_nodes ) :
    """
    Makes a neato-style undirected graph from the given corups.
    for m, the cocite matrix, m is upper triangular (and could be viewed as symmetric).  
    Entries represent the weight of the edge between row i and column j.

    This renders all nonzero edges, so clip first!  Edge thicknesses/weights are 
    determined by the bindings, while node sizes are determined by cv if given which is 
    a vector of total cites
    """

    O, row_dois, column_dois = cites_matrix( corpus )
    neato_cooccurrence_graph( O, column_dois )
    return None

    
    v = total_occurrences( O ) 
    nv = v.astype( float32 ) / v.max()
    C = cooccurrence_matrix ( O )
    nC = normalized_cooccurrence_matrix( O )

    # now find our cutoff!
    # find the max number of cocites and start there
    cocite_cutoff = C.max()
    num_nodes = nodes_from_c( C[C >= cocite_cutoff] )
    # then reduce the number until we exceed max_nodes
    while num_nodes < max_nodes :
        cocite_cutoff = cocite_cutoff - 1
        num_nodes = nodes_from_c( C[C >= cocite_cutoff] )

    if num_nodes > max_nodes :
        cocite_cutoff = cocite_cutoff + 1
        
    C = C.copy()
    C[ C < cocite_cutoff ]= 0

    graph = pydot.Dot( graph_type = 'graph' )
    graph.set_overlap("false")
    coords = zip(*(C >= cocite_cutoff).nonzero())

    # make a dict of all nodes which are mentioned in the coords
    nodes = {}
    index = 1
    for coord in set(chain.from_iterable(coords)) :
        if not nodes.has_key( coord ) :
            node =  pydot.Node( str(coord) )
            if v != None :
                doi = column_dois[coord]
                node.set_label( str(index) )
                node.set_penwidth( nv[ coord ] )
                node.set_fixedsize("true")
                node.set_width( 1.0 *nv[ coord ] )
                #node.set_shape("circle")
            nodes[ coord ] = node
            graph.add_node( node )
        index = index + 1

    for coord in coords :
        
        edge = pydot.Edge( nodes[coord[0]], nodes[coord[1]] )
        edge.set_weight( nC[coord] )
        edge.set_penwidth( nC[coord]*5 )
        #edge.set_label( str(int(m[coord]) ))
        graph.add_edge(edge)

   
    legend = pydot.Node( "legend" )
    nodelist = nodes.items()
    nodelist.sort( lambda a,b : cmp(node_index(a[1].get_label()),node_index(b[1].get_label())) )
    legend.set_label(  "\l".join([x[1].get_label()+":"+column_dois[x[0]] for x in nodelist])+"\l" )
    legend.set_shape("box")
    graph.add_node(legend)

    print graph.to_string()
    #graph.write_dot('test.dot', prog='neato' )
    #graph.write_png('test.png', prog='neato' )
    #graph.write_pdf('test.pdf', prog='neato' )


    


class Lsa_browser :
    def __init__( self, figure, pickdata ) :
        self.figure = figure
        self.pickdata = pickdata
        self.figure.canvas.mpl_connect('pick_event', self.on_pick )

    def on_pick( self, event ) :
        item = event.artist
        ind = event.ind
        print take( self.pickdata, ind )

def write_ggobi_lsa_csv( c, fnam, only_titles=True, max_dimensions = 100 ) :
    pickdata = [ x['DI']+': '+x['TI'] for x in c]
    if only_titles :
        O, rows, cols = title_words_matrix( c, False )
    else :
        O, rows, cols = abstract_words_matrix( c, False )
    O2, pickdata2, cols2 = prune_o(O, pickdata, cols)
    U,S,V = svd(tfidf_occurrence_matrix(O2), False)
    axes = range(1, min(max_dimensions, len(S)))
    with open( fnam, 'wb' ) as csvfile:
        writer = csv.writer( csvfile )
        writer.writerow( ["","Type"] + ["Axis "+str(x) for x in axes] )
        for x in range(U.shape[0]) :
            writer.writerow([pickdata2[x], "Doc"] + list(U[x]))
        for x in range(V.shape[1]) :
            writer.writerow([cols2[x], "Word"]+list(V[:,x]))

def lsa_graph( c, interactive = False, only_titles=True, x_axis = 1, y_axis = 2 ) :
    pickdata = [ x['DI']+': '+x['TI'] for x in c]
    if only_titles :
        O, rows, cols = title_words_matrix( c, False )
    else :
        O, rows, cols = abstract_words_matrix( c, False )
    O2, pickdata2, cols2 = prune_o(O, pickdata, cols)
    U,S,V = svd(tfidf_occurrence_matrix(O2), False)
    print S
    U2 = U[:,0:3]
    S2 = S[0:3]
    V2 = V[0:3,:]

    titles = plt.scatter( U2[:,x_axis], U2[:,y_axis], c='r', marker='^', picker = 5 )
    for x in range( U2.shape[0] ) :
        plt.annotate( str(x), (U2[x,x_axis],U2[x,y_axis]))
    plt.scatter( V2[x_axis,:],V2[y_axis,:], c='b', marker='o' )
    for x in range( V2.shape[1] ) :
        plt.annotate( cols2[x], (V2[x_axis,x],V2[y_axis,x]) )
    if interactive :
        browser = Lsa_browser( plt.gcf(), pickdata2 )
        #plt.gcf().canvas.mpl_connect('pick_event', on_lsa_pick)
        plt.show()
        plt.close()
    else :
        plt.savefig( "/tmp/test.png" )
        plt.close()

def test_lsa( interactive = False ) :
    #LSA introduction from http://www.puffinwarellc.com/index.php/news-and-articles/articles/33.html?showall=1
    LSA_test_titles = [ "The Neatest Little Guide to Stock Market Investing",
                        "Investing for Dummies, 4th Edition",
                        "The Little Book of Common Sense Investing: The Only Way To Guarantee Your Fair Share of Stock Market Returns",
                        "The Little Book of Value Investing",
                        "Value INvesting: From Graham to Buffett and Beyond",
                        "Rich Dad's Guide to Investing: What the Rich Invest In, That the Poor and the Middle Class Do Not",
                        "Investing in Real Estate, 5th Edition",
                        "Stock Investing for Dummies",
                        "Rich Dad's Advisors: The ABCs of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss" ]
    LSA_test_corpus = [ { 'TI' : x, 'DI' : x } for x in LSA_test_titles ]
    write_ggobi_lsa_csv( LSA_test_corpus, "/tmp/test.csv" )
    lsa_graph( LSA_test_corpus, interactive, x_axis = 1, y_axis = 2 )
    

c = import_wos("metamaterials_cited.tab")
O,rows,cols = abstract_words_matrix(c)
O2,rows2,cols2 = prune_o( O, rows, cols, 10 )
v = O2.sum(0)
cC = cosine_cooccurrence_matrix( O2 )
print cC.max()
iC = inclusion_index_cooccurrence_matrix( O2 )
#neato_cooccurrence_graph( cC, v, cols2, max_nodes = 10, fnam_stem="cosine", label_nodes_directly = False, scale=20.0*(cC.max() / 0.16))
neato_cooccurrence_graph( iC, v, cols2, max_nodes=10, fnam_stem="inclusion", label_nodes_directly = False, min_node_size=0.7, scale=3)
#col_labels = [[x['DI']+": "+x['TI'] for x in c if x['DI']==d][0] for d in cols2]
#neato_graph_from_corpus(c, 20)
#unittest.main()
