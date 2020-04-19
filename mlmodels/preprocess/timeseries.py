"""

  Ensemble of preprocessor for time series



https://docs-time.giotto.ai/


https://pypi.org/project/tslearn/#documentation


https://pypi.org/project/skits/



Gluon TS



"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict


from mlmodels.util import path_norm



##############################################################################################
def glutonts_to_pandas(dataset_name_list=["m4_hourly", "m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "m4_yearly", ]):
    """
     n general, the datasets provided by GluonTS are objects that consists of three main members:

dataset.train is an iterable collection of data entries used for training. Each entry corresponds to one time series
dataset.test is an iterable collection of data entries used for inference. The test dataset is an extended version of the train dataset that contains a window in the end of each time series that was not seen during training. This window has length equal to the recommended prediction length.
dataset.metadata contains metadata of the dataset such as the frequency of the time series, a recommended prediction horizon, associated features, etc.
In [5]:
entry = next(iter(dataset.train))
train_series = to_pandas(entry)
train_series.plot()

    # datasets = ["m4_hourly", "m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "m4_yearly", ]


    """
    from gluonts.dataset.repository.datasets import get_dataset
    from gluonts.dataset.util import to_pandas

    ds_dict = OrderedDict()
    for t in dataset_name_list :
      ds1 = get_dataset(t)
      
      ds_dict[t]['train'] = to_pandas( next(iter(ds1.train)) ) 
      ds_dict[t]['test'] = to_pandas( next(iter(ds1.test))  ) 
      ds_dict[t]['metadata'] = ds1.metadata 


    return ds_dict






def pandas_to_gluonts(df, pars=None) :
    """
    Custom datasets¶
    At this point, it is important to emphasize that GluonTS does not require this specific format for a custom dataset that a user may have. The only requirements for a custom dataset are to be iterable and have a “target” and a “start” field. To make this more clear, assume the common case where a dataset is in the form of a numpy.array and the index of the time series in a pandas.Timestamp (possibly different for each time series):

    In [8]:
    N = 10  # number of time series
    T = 100  # number of timesteps
    prediction_length = 24
    freq = "1H"
    custom_dataset = np.random.normal(size=(N, T))
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series
    Now, you can split your dataset and bring it in a GluonTS appropriate format with just two lines of code:

    In [9]:
    from gluonts.dataset.common import ListDataset
    In [10]:
    # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
    train_ds = ListDataset([{'target': x, 'start': start}
                            for x in custom_dataset[:, :-prediction_length]],
                           freq=freq)

    # test dataset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset([{'target': x, 'start': start}
                           for x in custom_dataset],
                          freq=freq)

        test_target_values = train_target_values.copy()
    train_target_values = [ts[:-single_prediction_length] for ts in train_df.values]

m5_dates = [pd.Timestamp("2011-01-29", freq='1D') for _ in range(len(sales_train_validation))]

train_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: fdr,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, start, fdr, fsc) in zip(train_target_values,
                                         m5_dates,
                                         train_cal_features_list,
                                         stat_cat)
], freq="D")



    """
            ### convert to gluont format
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName       

    



    ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: fdr,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, start, fdr, fsc) in zip(ytarget,
                                         dates,
                                         cols_dynamic,
                                         col_static)
    ], freq = pars['freq']


    ds = ListDataset([{FieldName.TARGET: df.iloc[i].values,  
        FieldName.START:  pars['start']}
                       for i in range(cols)], 
                       freq = pars['freq'])
    ds = None
    
    return ds 



class Preprocess_nbeats:

    def __init__(self,backcast_length, forecast_length):
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
    def compute(self,df):
        df = df.values  # just keep np array here for simplicity.
        norm_constant = np.max(df)
        df = df / norm_constant
        
        x_train_batch, y = [], []
        for i in range(self.backcast_length, len(df) - self.forecast_length):
            x_train_batch.append(df[i - self.backcast_length:i])
            y.append(df[i:i + self.forecast_length])
    
        x_train_batch = np.array(x_train_batch)[..., 0]
        y = np.array(y)[..., 0]
        self.data = x_train_batch,y
        
    def get_data(self):
        return self.data
        
class SklearnMinMaxScaler:

    def __init__(self, **args):
        self.preprocessor = MinMaxScaler(**args)
    def compute(self,df):
        self.preprocessor.fit(df)
        self.data = self.preprocessor.transform(df)
        
    def get_data(self):
        return self.data



def pd_load(path) :
   return pd.read_csv(path_norm(path ))



def pd_interpolate(df, cols, pars={"method": "linear", limit_area="inside"  }):
    """
        Series.interpolate(self, method='linear', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None, **kwargs)[source]¶
        Please note that only method='linear' is supported for DataFrame/Series with a MultiIndex.

        Parameters
        Interpolation technique to use. One of:

        ‘linear’: Ignore the index and treat the values as equally spaced. This is the only method supported on MultiIndexes.
        ‘time’: Works on daily and higher resolution data to interpolate given length of interval.
        ‘index’, ‘values’: use the actual numerical values of the index.
        ‘pad’: Fill in NaNs using existing values.
        ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’: Passed to scipy.interpolate.interp1d. These methods use the numerical values of the index. Both ‘polynomial’ and ‘spline’ require that you also specify an order (int), e.g. df.interpolate(method='polynomial', order=5).
        ‘krogh’, ‘piecewise_polynomial’, ‘spline’, ‘pchip’, ‘akima’: Wrappers around the SciPy interpolation methods of similar names. See Notes.
        ‘from_derivatives’: Refers to scipy.interpolate.BPoly.from_derivatives which replaces ‘piecewise_polynomial’ interpolation method in scipy 0.18.

        axis{0 or ‘index’, 1 or ‘columns’, None}, default None
        Axis to interpolate along.

        limitint, optional
        Maximum number of consecutive NaNs to fill. Must be greater than 0.

        inplacebool, default False
        Update the data in place if possible.

        limit_direction{‘forward’, ‘backward’, ‘both’}, default ‘forward’
        If limit is specified, consecutive NaNs will be filled in this direction.

        limit_area{None, ‘inside’, ‘outside’}, default None
        If limit is specified, consecutive NaNs will be filled with this restriction.
        None: No fill restriction.
        ‘inside’: Only fill NaNs surrounded by valid values (interpolate).
        ‘outside’: Only fill NaNs outside valid values (extrapolate).

        New in version 0.23.0.
        downcastoptional, ‘infer’ or None, defaults to None
        Downcast dtypes if possible.

        **kwargs
        Keyword arguments to pass on to the interpolating function.


    """
    for t in cols :
        df[t] = df[t].interpolate( **pars)

    return df



def pd_clean(df, cols=None,  pars={'na_value': 0.0 }) :
  cols = df.columns if cols is None else cols
  for t in cols :
    df = df.fillna( pars['na_value'])

  return df




def pd_fillna(df,**args):
    return df.fillna(**args)


def pd_reshape(test, features, target, pred_len, m_feat) :
    x_test = test[features]
    x_test = x_test.values.reshape(-1, pred_len, m_feat)
    y_test = test[target]
    y_test = y_test.values.reshape(-1, pred_len, 1)        
    return x_test, y_test



def time_train_test_split(data_pars):
    """
       train_data_path
       test_data_path
       predict_only

    """
    d = data_pars
    pred_len = d["prediction_length"]
    features = d["col_Xinput"]
    target   = d["col_ytarget"]
    m_feat   = len(features)


    # when train and test both are provided
    if d["test_data_path"]:
        test   = pd_load(d["test_data_path"])
        test   = pd_clean(test)
        x_test, y_test = pd_reshape(test, features, target, pred_len, m_feat) 
        if d["predict_only"]:
            return x_test, y_test


        train   = pd_load( d["train_data_path"])
        train   = pd_clean(train)
        x_train, y_train = pd_reshape(train, features, target, pred_len, m_feat) 

        return x_train, y_train, x_test, y_test
    


    # for when only train is provided
    df      = pd_load(d["train_data_path"])
    train   = df.iloc[:-pred_len]
    train   = pd_clean(train)
    x_train, y_train = pd_reshape(train, features, target, pred_len, m_feat) 


    test   = df.iloc[-pred_len:]
    test   = pd_clean(test)
    x_test, y_test = pd_reshape(test, features, target, pred_len, m_feat) 
    if d["predict_only"]:
        return x_test, y_test

    return x_train, y_train, x_test, y_test


