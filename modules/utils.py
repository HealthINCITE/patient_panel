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
    ir=df.pivot_table(index='id', columns='yrm', values=target, aggfunc='sum')
    print(ir.index)
    c=ir.columns
    results=pd.DataFrame() #final results

    pred_df=pd.read_csv(pred, index_col = 'id')
    pred_df=pred_df.sort_values(by=[target], ascending=False)
    pred_df=pred_df.iloc[0:capacity]
    ir=ir[ir.index.isin(pred_df.index)]
    pred_df=pred_df.sort_index()
    row=results.shape[0]
    pred_df['referral'] = np.where(pred_df[target] > threshold, 1, 0)
    print(pred_df)
    # Loop through the windows
    for w in windows:
        sl=slice(w[0],w[1])
        y= ir.iloc[:,sl].sum(axis=1) #take slice based on window
        print(y)
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

def preprocess_patients(df, date_col, date_format, con_start,con_months):
    """
    Preprocess patient data, looking for specific time windows.
    """
    df['date']=pd.to_datetime(df[date_col], format=date_format)
    temp_df=df.loc[:,['id','date']]
    temp_df=temp_df.loc[df['date']>=con_start]
    temp_df=temp_df.pivot_table(index='id',aggfunc='count')
    temp_df=temp_df[temp_df.date>=24]
    return df[df['id'].isin(temp_df.index)]

def preprocess_referrals(ref):
    """
    Preprocess referral data, aggregating classes and referrals.
    """
    ref['referrals']=1
    ref['date']=pd.to_datetime(ref['date'], format='%m/%d/%Y')
    ref['yrm']=ref['date'].dt.strftime('%Y%m').astype(int)
    mergedf=ref.loc[:,['id','yrm','referrals','class']]
    agg_ref = pd.pivot_table(mergedf, values='referrals', index=['id','yrm'],
                      aggfunc=np.sum)
    agg_class = pd.pivot_table(mergedf, values='class', index=['id','yrm'],
                      aggfunc=lambda x: ' '.join(x))
    agg_ref['referral']=1  # Add 1 for the month count.
    return  pd.merge(agg_ref, agg_class, on=['id', 'yrm'])

def merge_and_fill(pat, ref):
    """
    Merge patient and referral database.
    """
    df=pd.merge(pat, ref, how='left',  on=['id', 'yrm'])
    df['class']=df['class'].fillna('healthy')
    df['referral']=df['referral'].fillna(0).astype(int)
    df['referrals']=df['referrals'].fillna(0).astype(int)
    return df
