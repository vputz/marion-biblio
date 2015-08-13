from wos_reader import Wos_h5_reader
from pandas import DataFrame, Series
import pandas
from scipy.stats import norm, lognorm
from scipy.optimize import curve_fit, fmin, fsolve
from numpy import float64, log, exp, array, sqrt, diag, arange, linspace, random, pi

import numpy
import datetime

def cite_frame( series ) :
    return DataFrame( { 'cites' : series, 'cumsum' : series.cumsum() } )

#x = arange(0,5,0.1)
#y = norm.cdf(x)
#plot(x,y)
def cumcit( t, lamb, mu, sigma ) :
    m = 30.0
    #return m * (numpy.exp(lamb * lognorm.cdf( t, sigma, scale=mu ))-1.0 )
    return m*( numpy.exp( lamb * norm.cdf( (numpy.log(t)-mu)/sigma ) ) - 1.0 )

def cumcit_query( df, maxpoints=None, rescale_start=True, out_to=None, title="", only_title=False, use_mle = True, **kwargs ) :
    cumsum = df['cumsum'].dropna()
    x = array( cumsum.index, dtype=float64 )- ( cumsum.index[0] if rescale_start else 0 )
    y = array( cumsum, dtype=float64 )

    xplot = ( (x + df['cumsum'].index[0]) ) if rescale_start else x

    result = dict( xplot=xplot, yplot=y, errors=[] )
    
    #print "X: ", x
    if maxpoints :
        xfit = x[0:maxpoints]
        yfit = y[0:maxpoints]
    else :
        xfit = x
        yfit = y

    predict_x = linspace( 0, out_to if out_to else xplot[-1], 100 )
    result['predict_x'] = predict_x

        # now try the basic curve fit
    try :
        popt, pcov = curve_fit( cumcit, xfit, yfit, **kwargs )

        devs = sqrt(diag(pcov))
        popt_fit_y = cumcit( predict_x, *popt )
        popt_high_y = cumcit( predict_x, *(popt+devs) )
        popt_low_y = cumcit( predict_x, *(popt-devs) )
        
        result['popt'] = popt
        result['pdevs'] = devs
        result['popt_fit_y'] = popt_fit_y
        result['popt_high_y'] = popt_high_y
        result['popt_low_y'] = popt_low_y

    except RuntimeError as e:

        result['errors'].append( str(e) )

    try :
        offset = 0 if xfit[0] != 0 else 0.1
        
        mle_opt = parms_from_likelihood( df, offset )
        mle_x = x.copy()
        mle_y = cumcit( x, *mle_opt[1] )

        result['mle_x'] = mle_x
        result['mle_y'] = mle_y
        result['mle_opt'] = mle_opt

    except RuntimeError as e :

        result['errors'].append( str(e) )

    return result

def plot_prediction( df, title=None, out_to=100, inset=[0.5,0.3,0.3,0.3], **kwargs ) :
    plot_cumcit( df, out_to=out_to, title=title, **kwargs )
    a = pylab.axes( inset )
    plot_cumcit( df, **kwargs )

def mu_day_to_year( mu ) :
    return log( exp( mu ) / 365 )

months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
def month_intervals( s, incr ) :
    result = {}
    for i in range(len(s) - incr + 1) :
        result["-".join((s[i], s[i+incr-1]))]= s[i+incr-1]
    return result

month_replacements = dict( list(month_intervals( months, 2 ).items()) 
                           + list(month_intervals(months,3).items())
                           + list(month_intervals(months,4).items())
                           + list(month_intervals(months,5).items())
                           + list(month_intervals(months,6).items())
                           )

# now we need to convert the dates into numbers, preferably fractional years
def date_to_number( s, randomize_ambiguity = False ) :
    for k in month_replacements.keys() :
        s = s.replace(k,month_replacements[k]).strip()
    try :
        return datetime.datetime.strptime( s, "%b %d %Y" ).toordinal()
    except ValueError :
        try :
            return datetime.datetime.strptime( s, "%b %Y" ).toordinal() + (random.randint(0,30) if randomize_ambiguity else 0)
        except ValueError :
            try :
                return datetime.datetime.strptime( s, "%Y" ).toordinal() + (random.randint(0,365) if randomize_ambiguity else 0)
            except ValueError :
                print("Error: unable to convert '{s}'".format( s=s))
                return numpy.nan

def date_to_year( s, randomize_ambiguity = False ) :
    return date_to_number(s, randomize_ambiguity ) / 365.

#print date_to_number( "MAR 2010" )
#print date_to_number( "MAR-APR 2010" )
#print date_to_number( "MAR 15 2010" )
def dates_from_wos_begin( w5, randomize_ambiguity = False ) :
    days = [ date_to_number(x['pubdate']+" "+str(x['pubyear']), randomize_ambiguity) for x in w5.h5.root.papers]
    minday = min(days)
    result = (array(days, float64) - minday)/365
    return result


def cite_frame_from_wos( wos_fnam, pubdate= None, randomize_ambiguity=False, max_time=None, trim_end=None ) :
    wos = Wos_h5_reader( wos_fnam )
    def fix_spans( d ) :
        
        for k in month_replacements.keys() :
            d = d.replace(k,month_replacements[k])
        return d
    dates = [ " ".join( x ) for x in zip((row['pubdate'].decode('utf-8') for row in wos.h5.root.papers),
                                         (str(row['pubyear']) for row in wos.h5.root.papers)) ]
    # there's a chance that pubyear could be -1 if the original data
    # didn't have a number for pubyear, which would throw data in error; not sure how to handle yet
    cites = Series( [1]*len(dates) )
    #sdates = Series([pandas.lib.Timestamp(fix_spans(x)) for x in dates] )
    sdates = Series( [date_to_number(fix_spans(x), randomize_ambiguity ) for x in dates] )
    df=DataFrame( { 'dates' : sdates, 'cites' : cites } )
    df2=df.groupby('dates').sum()
    if ( pubdate == None ) :
        start_date = df2.index[0]
    else :
        start_date = pandas.lib.Timestamp( pubdate )
    df2.index = Series(df2.index).apply( lambda x : float((x - start_date))/365 )
    #df2.index = Series(df2.index).apply( lambda x : float((x - start_date).days)/365 )
    df2['cumsum'] = df2.cites.cumsum()
    if max_time :
        return df2[df2.index < max_time]
    if trim_end :
        return df2[df2.index < max(df2.index) - trim_end]
    return df2

def infcit_from_lambda( lam ) :
    m=30
    return int((m*(numpy.exp(lam) - 1)).round() )

def infcit( df, **kwargs ) :
    # calculates the guess of total infinite citations; requires lambda from 
    # the cumulative citations curve fit
    x = array( df['cumsum'].index, dtype=float64 )- df['cumsum'].index[0]
    y = array( df['cumsum'], dtype=float64 )
    popt, pcov = curve_fit( cumcit, x, y, maxfev=100*len(x), **kwargs )
    lam = popt[0]
    var = sqrt(pcov[0,0])
    # we actually don't care much except for popt[0], which is lambda
    m = 30
    return infcit_from_lambda( lam ), infcit_from_lambda( lam+var ), infcit_from_lambda( lam-var )

#isolating these two to try and understand the difficulties with the MLE
def norm_pdf( T, mu, sigma ) : #"Pg" in paper
    x = (log(T)-mu)/sigma
    #return 1.0/sqrt(2*pi) * exp( -(x**2)/2 )
    return norm.pdf(x)
    #return norm.pdf( (log(T)-mu)/sigma )

def lognorm_pdf( t, mu, sigma ) :
    return 1.0 / (sqrt(2*pi)*sigma*t) * exp( - (log(t)-mu)**2 / (2*sigma**2) )

def lognorm_cdf( T, mu, sigma ) : #"phi" in paper
    #return lognorm.cdf( T, sigma, scale=mu )
    #return lognorm.cdf( (log(T)-mu)/sigma ) # fairly close?
    return norm.cdf( (log(T)-mu)/sigma )
#
# MLE routines for estimation
def nlog_likelihood( x, ts ) :
    # x -> [lambda, mu, sigma]
    # args =[ array_of_times, see obslist ]
    lamb = x[0]
    mu = x[1]
    sigma = x[2]
    #ts = args[0]#numpy.array( df.index, dtype= float64 )
    #print args
    N = ts.shape[0]
    m = 30
    T = ts[-1]
    sum1 = log( arange( 1, N+1, dtype=float64 ) + m - 1 ).sum()
    #sum1 =array([ log(i + m - 1) for i in range( 1, N ) ], float64).sum()
    #test_a = array([ ts[i] for i in range( 1, N ) ], float64 )
    #sum2 = array([ lognorm.logpdf( ts[i], sigma, scale=mu ) for i in range( 1, N ) ], float64).sum()
    sum2 = log( lognorm_pdf( ts, mu, sigma ) ).sum()#lognorm.logpdf( ts, sigma, scale=mu ).sum()
    #sum3 = array([ lognorm.cdf( ts[i], sigma, mu ) for i in range( 1, N ) ], float64).sum() 
    sum3 = lognorm_cdf( ts, mu, sigma ).sum()
    return -(N*log(lamb) + sum1 + sum2 - lamb*(N+m)*lognorm_cdf( T, mu, sigma ) + lamb*sum3)

def nlog_likelihat( x, ts ) :
    lamb = x[0]
    mu = x[1]
    sigma = x[2]
    N = ts.shape[0] #ignore first ("zero") observation?
    m = 30.0
    T = ts[-1]
    mhat = m/(N)
    mean1 = log( arange(0,N, dtype=float64)/N + mhat ).mean()
    #print "Mean1: {mean1}".format(mean1=mean1)
    mean2 = ( log(lognorm_pdf(ts, mu, sigma)) + lamb * lognorm_cdf( ts, mu, sigma ) ).mean()
    #print mean2
    return -( log(lamb) + mean1 + mean2 - lamb*(1+mhat)*lognorm_cdf( T, mu, sigma ) )

import itertools

def obslist( df ) :
    # takes a citeframe with series (time, num_obs) and converts it to a string of
    # observations so [ (11,1), (13, 2)] -> [11,13,13]
    return array( [x for x in itertools.chain.from_iterable([ [df.index[x]] * df.cites.iloc[x] for x in range( len(df.index) ) ])] )

def tval( t, mu, sigma ) :
    return (log(t)-mu)/sigma

def s25( x, ts ) :
    # x[0]= lambda, x[1]=mu, x[2]= sigma
    # args[0] = T, args[1] is citeframe containing observations
    lamb = x[0]
    mu = x[1]
    sigma = x[2]
    N = ts.shape[0]
    mhat = 30.0/N
    T = ts[-1]
    mean1 = lognorm_cdf( ts, mu, sigma ).mean()#lognorm.cdf( ts, sigma, scale=mu ).mean()
    return lamb * ( (1 + mhat)*lognorm_cdf( T, mu, sigma ) - mean1 ) - 1.0

def s26a( x, ts ) :
    lamb = x[0]
    mu = log(x[1])
    sigma = x[2]
    N = ts.shape[0]
    mhat = 30.0/N
    T = ts[-1]
    mean1 = ( tval(ts, mu, sigma ) - lamb*norm_pdf( ts, mu, sigma )).mean() #(log(ts)-mu)/sigma ) ).mean()
    return mean1 + lamb*(1+mhat)*norm_pdf( T, mu, sigma )#(log(T)-mu)/sigma )

def s26b( x, ts ) :
    lamb = x[0]
    mu = log(x[1])
    sigma = x[2]
    N = ts.shape[0]
    mhat = 30.0/N
    T = ts[-1]
    mean1 = ( tval(ts, mu, sigma)*( tval(ts,mu,sigma) - lamb*norm_pdf( ts, mu, sigma ) ) ).mean()
    return mean1 + lamb*(1 + mhat)*tval(T, mu, sigma )*norm_pdf( T, mu, sigma ) - 1.0

def mleeqn( x, ts ) :
    return array([s25( x, ts ), s26a( x, ts ), s26b( x, ts )])

def parms_from_likelihood( df, offset = 0.0, initial_guess = [1.0,1.0,1.0] ) :
    ts = obslist( df ) + offset
    # tuple([ts]) notation seems to be necessary to avoid converting ts to a long tuple itself
    fminhood = fmin( nlog_likelihood_min, initial_guess, args=tuple([ts]) )
    fminhat  = fmin( nlog_likelihat, initial_guess, args=tuple([ts]) )
    mle = fsolve( mleeqn, fminhood, args=ts )
    #return fminhat
    return (fminhood, fminhat, mle)

    
