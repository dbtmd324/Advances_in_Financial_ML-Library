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
    # thres > 0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = w[-1] / k * (d-k+1)
        w.append(w_)

    w = np.array(w[::-1]).reshape(-1, 1)

    return w

def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')

    ax = w.plot()
    ax.legend(loc='upper left');mpl.show()

    return

if __name__ == '__main__':
    plotWeights(dRange=[0,1], nPlots=11, size=6)
    plotWeights(dRange=[1,2], nPlots=11, size=6)

#### SNIPPET 5.2: STANDARD FRACDIFF (EXPANDING WINDOW) ####

def fracDiff(series, d, thres=0.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, noting is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    # 1. Compute weights for the longest series
    w = getWeights_FFD(d, series.shape[0])
    # 2. Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_/= w_[-1]
    skip = w_[w_>thres].shape[0]
    # 3. Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
               continue # exclude NAs
            df_[loc] = np.dot(w[-(iloc+1):, :].T, seriesF.loc[:loc])[0,0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)

    return df 

#### SNIPPET 5.3 THE NEW FIXED-WIDTH WINDOW FRACDIFF METHOD ####

def fracDiff_FFD(Series, d, thres=1e-5):
    '''
    Constant width window(new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be positive fractional, not necessarily bounded [0,1].
    '''
    # 1. Compute weights for the longest series
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    # 2. Apply weights to values
    df = {}

    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
               continue # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]
        df[name] = df_.copy(deep=True)

    df = pd.concat(df, axis=1)

    return df
    
#### SNIPPET 5.4 FINDING THE MINIMUM D VALUE THAT PASSES THE ADF TEST ####

def plotMinFFD():
	from statsmodels.tsa.stattools import adfuller
	path, instName = './', 'ES1_Index_Method12'
	out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
	df0 = pd.read_csv(path + instName + '.csv', index_col=0, parse_dates=True)
	for d in np.linspace(0, 1, 11):
		df1 = np.log(df0[['Close']]).resample('1D').last() # downcast to daily obs
    df2 = fracDiff_FFD(df1, d, thres=0.01)
    corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2['Close'])[0, 1]
		df2 = adfuller(df2['Close'], maxlag=1, regression='c', autolag=None)
		out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr] # with critical value
	out.to_csv(path + instName + '_testMinFFD.csv')
	out[['adfStat', 'corr']].plot(secondary_y='adfStat')
	mpl.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
	mpl.savefig(path+instName+'_testMinFFD.png')

	return

#### SNIPPET 6.1: ACCURACY OF THE BAGGING CLASSIFIER ####

from scipy.misc import comb
N, p, k = 100, 1.0/3, 3.0
p_ = 0

for i in xrange(0, int(N/k)+1):
	p_ += comb(N, i) * p**i * (1-p)**(N-i)
print p, 1-p

#### SNIPPET 6.2: 3 WAYS OF SETTING UP AN RF ####

clf0 = RandomForestClassifier(n_estimators=1000, class_weight='balanced_subsample', criterion='entropy')
clf1 = RandomForestClassifier(criterion='entropy', max_features='auto', class_weight='balanced')
clf1 = BaggingClassifier(base_estimator=clf1, n_estimators=1000, max_samples=avgU)
clf2 = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
clf2 = RandomForestClassifier(base_estimator=clf2, n_estimators=1000, max_samples=avgU, max_features=1)

#### SNIPPET 7.1: PURGING OBSERVATION IN THE TRAINING SET ####

def getTrainTimes(t1, testTimes):
	'''
	Given testTimes, find the times of the training observations.
   - t1.index: Time when the observation started.
   - t1.value: Time when the observation ended.
   - testTimes: Times of testing observations.
	'''
	trn = t1.copy(deep=True)
  for i,j in testTimes.iteritems():
		df0 = trn[(i<=trn.index)&(trn.index<=j)].index # train starts within test
		df1 = trn[(i<=trn)&(trn<=j)].index # train ends within test
	  df2 = trn[(trn.index<=i)&(j<=trn)].index # train envelops test
		trn = trn.drop(df0.union(df1).union(df2))

	return trn