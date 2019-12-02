import sys
import pandas as pd
import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, balanced_accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import roc_curve

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def merge(per_df, per_col, per_date_col, per_date_format, ref_df,ref_date_col, ref_date_format, ref_shift_months):
    #start by preprocessing referrals
    
    ref_df=preprocess_referrals(ref_df, per_col, ref_date_col, ref_date_format, ref_shift_months)
    #preprocess people 
    per_df=preprocess_persons(per_df, per_col, per_date_col, per_date_format)
    
    #Finalmerge
    df=pd.merge(per_df, ref_df, how='left',  on=[per_col, per_date_col])
    df=fill_na(df,['lab_','ref_'],0, int)
    df=fill_na(df,['class'],'h',str)
    df=fill_na(df,['labels'],'0',str)

    #df.set_index(['person_id','myr'])
    #bwd = df[['person_id']+['ref']].sort_index().groupby("person_id")#.rolling(3, min_periods=1).sum().astype(int)
    #bwd.drop(['person_id'], axis=1,inplace=True)
    #bwd.reset_index(inplace=True)
    #bwd.rename(columns = {'ref':'recent_ref'}, inplace = True)
    # There is create a recent. 
    #bwd.columns['ref']=bwd.columns['recent_ref']
    #df=pd.merge(df, bwd)
    #df.drop(['level_1'], axis=1,inplace=True)
    return df

def fill_na(df, patterns, value, c_type):
    for pattern in patterns:
        cols=df.columns[df.columns.str.contains(pattern)]
        for x in cols:
            df[x]=df[x].fillna(value).astype(c_type)
    return df

def preprocess_persons(df, person_col, date_col, date_format):
    """
    Preprocess patient data, looking for specific time windows.
    """
    df.columns=map(str.lower,df.columns)
    df['date']=pd.to_datetime(df[date_col], format=date_format)
    return df
    #return temp_df

def preprocess_referrals(ref, per_col, date_col, date_format,shift_months):
    """
    Preprocess referral data, aggregating classes and referrals.
    """
    ref['date']=pd.to_datetime(ref[date_col], format= date_format)
    if shift_months != 0 :
        ref['date']=ref['date'] + pd.DateOffset(months=shift_months)
    #Create the Year/Month format used in
    ref['myr']=ref['date'].dt.strftime('%Y%m').astype(int)

    mergedf=ref.loc[:,[per_col,'myr','class']]
    trans=ref['class'].unique()
    translate=dict(zip(ref['class'].unique(),[x for x in range(1,len(trans)+1)]))
    mergedf['labels']=ref['class'].map(lambda x: translate[x])
    lab_dum=pd.get_dummies(mergedf['labels'],prefix='lab')
    mergedf= pd.concat([mergedf, lab_dum], axis=1);
    agg_ref = pd.pivot_table(mergedf, values=lab_dum.columns, index=[per_col,'myr'], aggfunc=np.sum)
    agg_ref['ref_m']=agg_ref.sum(axis=1)
    agg_ref['ref_']=1
    
    #This aggregates all the reasons in the class. 
    #TBD sort this or one hot encode
    mergedf['labels']= mergedf['labels'].astype(str)
    agg_class = pd.pivot_table(mergedf, values=['class','labels'], index=[per_col,'myr'], aggfunc=lambda x: ', '.join(x))
    df= pd.merge(agg_ref, agg_class, on=[per_col, 'myr'])
    df.reset_index(inplace=True)
    df['ref_c']=df.groupby([per_col])['ref_m'].cumsum()
    return df


#Define the function
def train_test_split(df, date_col, date_format, split_time):
    """
    Provide an train/test split based on a timestamp.
    df = Dataframe (Pandas dataframe).
    date_col = Date column (string).
    date_format = The date format.
    split_time = A specific place to date. (date format)
    """
    split =pd.Timestamp(split_time)
    #Let's convert this to datetime while we are at it.
    df['yrm'] = pd.to_datetime(df[date_col], format=date_format)
    train=df.loc[df['yrm']<split]
    test=df.loc[df['yrm']>split]
    return train, test

def score_windows(exp, df, pred, capacity, windows, results_file,  target, threshold, save=True, append=True):
    """
    exp = Experiment name.
    df = evaluation
    capacity = Capacity

    """
    ir=df.pivot_table(index='person_id', columns='yrm', values=target, aggfunc='sum')
    c=ir.columns
    results=pd.DataFrame() #final results

    pred_df=pd.read_csv(pred, index_col = 'person_id')
    pred_df=pred_df.sort_values(by=[target], ascending=False)
    pred_df=pred_df.iloc[0:capacity]
    ir=ir[ir.index.isin(pred_df.index)]
    pred_df=pred_df.sort_index()
    row=results.shape[0]
    pred_df['referral'] = np.where(pred_df[target] > threshold, 1, 0)
    # Loop through the windows
    for w in windows:
        sl=slice(w[0],w[1])
        y= ir.iloc[:,sl].sum(axis=1) #take slice based on window
        label=c[w[0]].strftime('%Y%m')+'-'+c[w[1]-1].strftime('%Y%m')
        results.loc[row, 'experiment']=exp
        results.loc[row, 'date']=pd.Timestamp.now(tz=None)
        results.loc[row, 'range']=label
        results.loc[row, 'log_loss'] = log_loss(y, pred_df[target])

        results.loc[row, 'precision']=precision_score(y, pred_df['referral'])
        results.loc[row, 'recall']=recall_score(y, pred_df['referral'])
        results.loc[row, 'accuracy']=accuracy_score(y, pred_df['referral'])
        results.loc[row, 'balanced_accuracy']=balanced_accuracy_score(y, pred_df['referral'])
        results.loc[row, 'f1']=f1_score(y, pred_df['referral'])
        row=row+1
    results.to_csv(results_file, index = False)
    return results

def select_window(df, con_start, con_months):
    temp_df=df.loc[:,[person_col,'date']]
    temp_df=temp_df.loc[temp_df['date']>=con_start]
    temp_df=temp_df.pivot_table(index=person_col,aggfunc='count')
    temp_df=temp_df[temp_df['date']>=24]
    return df[df[person_col].isin(temp_df.index)]



def merge_and_fill(pat, ref, person_col, date_col,fill_na):
    """
    Merge patient and referral database.
    """
    df=pd.merge(pat, ref, how='left',  on=[person_col, date_col])
    df['class']=df['class'].fillna('healthy')
    for x in fill_na:
        df[x]=df[x].fillna(0).astype(int)

    df.set_index(['person_id','myr'])
    bwd = df[['person_id']+['ref']].sort_index().groupby("person_id").rolling(3, min_periods=1).sum().astype(int)
    bwd.drop(['person_id'], axis=1,inplace=True)
    bwd.reset_index(inplace=True)
    bwd.rename(columns = {'ref':'recent_ref'}, inplace = True)
    # There is create a recent. 
    #bwd.columns['ref']=bwd.columns['recent_ref']
    #df=pd.merge(df, bwd)
    #df.drop(['level_1'], axis=1,inplace=True)
    return df


