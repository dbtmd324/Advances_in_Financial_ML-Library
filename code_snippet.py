"""
CODE SNIPPET OF ADVANCES FINANCIAL MACHINE LEARNING
"""

import numpy as np
import pandas as pd

#### SNIPPET 3.1 DAILY VOLATILITY ESTIMATES ####
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False):
    # 1. get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt>minRet] # minRet
    # 2. get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3. form events object, apply stop loss on t1
    side_ = pd.Series(1., index=trgt.index)
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_},
                       axis=1).dropna(subset=['trgt'])
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads,
                      close=close, events=events, ptSl=[ptSl, ptSl])
    events['t1'] = df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    events = events.drop('side', axis=1)

    return events

#### SNIPPET 3.2 TRIPLE-BARRIER LABELLING METHOD ####
def applyPtSlOnT1(close, events, ptSl, molecule):
	# apply stop loss/profit taking, if it takes place before t1 (end of event)
	events_=events.loc[molecule]
	out=events_[['t1']].copy(deep=True)

	if ptSl[0]>0: 
		pt=ptSl[0] * events_['trgt']
	else:
		pt=pd.Series(index=events.index) # NaNs
	
	if ptSl[1]>0:
		sl=-ptSl[1] * event_['trgt']
	else:
		sl=pd.Series(index=events.index) # NaNs

	for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
		df0=close[loc:t1] # path prices
		df0=(df0/close[loc] - 1) * events_.at[loc, 'side'] # path returns
		out.loc[loc, 'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss
		out.loc[loc, 'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking

	return out

#### SNIPPET 3.3 GETTING THE TIME OF FRIST TOUCH ####
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False):
    # 1. get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt>minRet] # minRet
    # 2. get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3. form events object, apply stop loss on t1
    side_ = pd.Series(1., index=trgt.index)
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_},
                       axis=1).dropna(subset=['trgt'])
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads,
                      close=close, events=events, ptSl=[ptSl, ptSl])
    events['t1'] = df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    events = events.drop('side', axis=1)

    return events
```
#### SNIPPET 3.4: ADDING A VERTICAL BARRIER ####
t1 = close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
t1 = t1[t1<close.shape[0]]
t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]) # NaNs at end

#### SNIPPET 3.5: LABELLING FOR SIDE AND SIZE ####
def getBins(events, close):
    # 1. prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2. create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[evenets_.index] - 1
    out['bin'] = np.sign(out['ret'])
    
    return out

#### SNIPPET 3.6: EXPANDING getEvents TO INCORPORATE META-LABELING ####
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1. get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt>minRet] # minRet
    # 2. get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3. form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads,
                      close=inst['Close'], events=events, ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    if side is None:
        events.drop('side', axis=1)
    
    return events

#### SNIPPET 3.7: EXPANDING getBins TO INCORPORATE META-LABELING ####
def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    - events.index is event's starttime
    - events['t1'] is event's endtime
    - event['trgt'] is event's target
    - events['side] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1, 1) <- label by price action
    Case 2: ('side' in events): bin in (0, 1) <- label by pnl (meta-labeling)
    '''
    # 1. prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2. create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side'] # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret']<=0, 'bin'] = 0 # meta-labeling
    
    return out

#### SNIPPET 3.8: DROPPING UNDER-POPULATED LABELS ####
def dropLabels(events, minPct=.05):
	# apply weights, drop labels with insufficient examples
	while True:
		df0 = events['bin'].value_counts(normalize=True)
		if df0.min() > minPct or df0.shape[0] < 3:
			break
		print ('dropped label', df0.argmin(), df0.min())
		events = events[events['bin'] != df0.argmin()]
	return events

#### SNIPPET 4.1: ESTIMATING THE UNIQUENESS OF A LABEL ####
def mpNumCoEvents(closeIdx, t1, molecule):
	'''
	Compute the number of concurrent events per bar.
	+molecule[0] is the date of the first event on which the weight will be computed
	+molecule[-1] is the date of the last event on which the weight will be computed
	'''

#### SNIPPET 4.2: ESTIMATING THE AVERAGE UNIQUENESS OF A LABEL ####
def mpSampleTW(t1, numCoEvents, molecule):
	# Derive average uniqueness over the event's lifespan
	wght = pd.Series(index=molecule)

	for tIn, tOut in t1.loc[wght.index].iteritems():
		wght.loc[tIn]=(1. / numCoEvents.loc[tIn:tOut]).mean()
	
	return wght

#### SNIPPPET 4.3: BUILD AN INDICATOR MATRIX ####
def getIndMatrix(barIx, t1):
	# Get indicator matrix
	indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
	
	for i, (t0, t1) in enumerate(t1.iteritems()):
		indM.loc[t0:t1, i]=1

	return indM

#### SNIPPET 4.4: COMPUTE AVERAGE UNIQUENESS ####
def getAvgUniqueness(indM):
	# Average uniqueness from indicator matrix
    c = indM.sum(axis=1) # concurrency
    u = indM.div(c, axis=0) # uniqueness
    avgU = u[u>0].mean() # average uniqueness

    return avgU

#### SNIPPET 4.5: RETURN SAMPLE FROM SEQUENTIAL BOOTSTRAP ####
def seqBootstrap(indM, sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:
        sLength = indM.shape[1]
    
    phi = []

    while len(phi) < sLength:
        avgU = pd.Series()

        for i in indM:
            indM_ = indM[phi+[i]] # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]

        prob = avgU / avgU.sum() # draw prob
        phi += [np.random.choice(indM.columns, p=prob)]

    return phi
        
#### SNIPPET 4.6 EXAMPLE OF SEQUENTIAL BOOTSTRAP ####

def main():
	t1 = pd.Series([2, 3, 5], index=[0, 2, 4]) # t0, t1 for each feature observation
	barIx = range(t1.max() + 1) # idnex or bars
	indM = getIndMatrix(barIx, t1)
	phi = np.random.choice(indM.columns, size=indM.shape[1])
	
	print(phi)
	print('Standard uniqueness: ', getAvgUniqueness(indM[phi]).mean())

	phi = seqBootstrap(indM)

	print(phi)
	print('Sequential uniqueness: ', getAvgUniqueness(indM[phi]).mean())

	return

#### SNIPPET 4.7: GENERATING A RANDOM T1 SERIES ####

def getRndT1(numObs, numBars, maxH):
    # random t1 Series
    t1 = pd.Series()

    for i in xrange(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
    
    return t1.sort_index()

#### SNIPPET 4.8: UNIQUENESS FROM STANDARD AND SEQUENTIAL BOOTSTRAP ALGORITHM ####

def auxMC(numObs, numBars, maxH):
    # Parallelized auxiliary function
    t1 = getRndT1(numObs, numBars, maxH)
    barIx = range(t1.max()+1)
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    
    return {'stdU': stdU, 'seqU': seqU}

#### SNIPPET 4.9: MULTI-THREADED MONTE CARLO ####

from mpEngine import processJobs, processJobs_

def mainMC(numObs=10, numBars=100, maxH=5, numIter=1E6, numThreads=24):
    # Monte Carlo experiments
    jobs= []

    for i in xrange(int(numIters)):
        job = {'func': auxMC, 'numObs': numObs, 'numBars': numBars, 'maxH': maxH}
        jobs.append(job)

    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)

    print (pd.DataFrame(out).describe())

    return

#### SNIPPET 4.10: DETERINATION OF SAMPLE WEIGHT BY ABSOLUTE RETURN ATTRIBUTION ####

def mpSampleW(t1, numCoEvents, close, molecule):
    # Derive sample weight by return attribution
    ret = np.log(close).diff() # log-returns, so that they are additive
    wght = pd.Series(index=molecule)

    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (ret.loc[tIn: tOut] / numCoEvents.loc[tIn: tOut]).sum()
    
    return wght.abs()

out['w'] = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1 = events['t1'], numCoEvents = numCoEvents, close=close)
out['w'] = out.shape[0] / out['w'].sum()

#### SNIPPET 4.11: IMPLEMENTATION OF TIME-DECAY FACTORS ####

def getTimeDecay(tW, clfLastW=1):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = tW.sort_index().cumsum()

    if clfLastW >= 0:
        slope = (1. - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1. / ((clfLastW+1) * clfW.iloc[-1])
    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW<0] = 0

    print(const, slope)

    return clfW

#### SNIPPET 5.1: WEIGHTING FUNCTION ####
def getWeights(d, size):
     