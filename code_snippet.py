"""
CODE SNIPPET OF ADVANCES FINANCIAL MACHINE LEARNING
"""

#### SNIPPET 3.1 ####

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

t1 = close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
t1 = t1[t1<close.shape[0]]
t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]) # NaNs at end

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