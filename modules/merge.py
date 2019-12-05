import sys
import pandas as pd
import datetime
from pathlib import Path
import numpy as np

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def merge(per_df, per_col, per_date_col, per_date_format, ref_df,ref_date_col, ref_date_format, gen_features=True):
    #start by preprocessing referrals
    
    ref_df, translate=preprocess_referrals(ref_df, per_col, ref_date_col, ref_date_format)
    #preprocess people 
    per_df=preprocess_persons(per_df, per_col, per_date_col, per_date_format)
    
    #Finalmerge
    df=pd.merge(per_df, ref_df, how='left',  on=[per_col, per_date_col])
    df=fill_na(df,['lab_','ref'],0, int)
    df=fill_na(df,['class'],'h',str)
    df=fill_na(df,['labels'],'0',str)
    if gen_features==True:
        df=generate_features(df, per_col, ['lab_'], 'lag_1')
        df=fill_na(df,['lag1_lab'],'0',int)
        
        df=generate_features(df, per_col, ['lag1_lab'], 'sum')
        df=generate_features(df, per_col, ['lag1_lab'], 'win_6')
    return df, translate

def fill_na(df, patterns, value, c_type):
    for pattern in patterns:
        cols=df.columns[df.columns.str.contains(pattern)]
        for x in cols:
            df[x]=df[x].fillna(value).astype(c_type)
    return df

def generate_features(df, per_col, patterns, tag, units=0):
    for pattern in patterns:
        cols=df.columns[df.columns.str[0:len(pattern)]==pattern]
        for x in cols:
            if tag[0:3]=='sum':
                df[tag+'_'+x[-5:]]=df.groupby([per_col])[x].cumsum()
            elif tag[0:3]=='win':
                num=int(tag.split('_')[1])
                df['win'+str(num)+'_'+x[-5:]]=df[['person_id',x]].sort_index().groupby(per_col).rolling(num, min_periods=1).sum().astype(int)[x].to_numpy(dtype=int)
            elif tag[0:3]=='lag':
                num=int(tag.split('_')[1])
                df['lag'+str(num)+'_'+x]=df[['person_id',x]].groupby(per_col).shift(num)
    return df

def preprocess_persons(df, person_col, date_col, date_format):
    """
    Preprocess patient data, looking for specific time windows.
    """
    df.columns=map(str.lower,df.columns)
    df['date']=pd.to_datetime(df[date_col], format=date_format)
    return df
    #return temp_df

def preprocess_referrals(ref, per_col, date_col, date_format):
    """
    Preprocess referral data, aggregating classes and referrals.
    """
    ref['date']=pd.to_datetime(ref[date_col], format= date_format)
    #if shift_months != 0 :
    #    ref['date']=ref['date'] + pd.DateOffset(months=shift_months)
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
    agg_ref['ref']=1
    
    #This aggregates all the reasons in the class. 
    #TBD sort this or one hot encode
    mergedf['labels']= mergedf['labels'].astype(str)
    agg_class = pd.pivot_table(mergedf, values=['class','labels'], index=[per_col,'myr'], aggfunc=lambda x: ', '.join(x))
    df= pd.merge(agg_ref, agg_class, on=[per_col, 'myr'])
    df.reset_index(inplace=True)
    return df, pd.DataFrame.from_dict(translate, orient='index')
