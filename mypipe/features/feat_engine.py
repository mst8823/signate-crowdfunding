import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import xfeat
from mypipe.features.base import BaseBlock


class OrdinalEncodingBlock(BaseBlock):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = None

    def fit(self, input_df, y=None):
        self.encoder = ce.OrdinalEncoder()
        self.encoder.fit(input_df[self.cols])

    def transform(self, input_df):
        return self.encoder.transform(input_df[self.cols]).add_prefix("OE_")


class CountEncodingBlock(BaseBlock):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = None

    def fit(self, input_df, y=None):
        self.encoder = ce.CountEncoder()
        self.encoder.fit(input_df[self.cols])

    def transform(self, input_df):
        return self.encoder.transform(input_df[self.cols]).add_prefix("CE_")


class OneHotEncodingBlock(BaseBlock):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = None

    def fit(self, input_df, y=None):
        self.encoder = ce.OneHotEncoder(use_cat_names=True)
        self.encoder.fit(input_df[self.cols])

    def transform(self, input_df):
        return self.encoder.transform(input_df[self.cols]).add_prefix("OHE_")


class GroupingBlock(BaseBlock): 
    """
    refer to xfeat.aggregation
    """

    def __init__(self, group_key, group_values, agg_methods):
        self.group_key = group_key
        self.group_values = group_values

        ex_agg_methods = ["max-min", "min-mean", "max-mean"]
        ex_trans_methods = ["val-mean", "z-score"]
        self.ex_agg_methods = [m for m in agg_methods if m in ex_agg_methods]
        self.ex_trans_methods = [m for m in agg_methods if m in ex_trans_methods]
        self.agg_methods = [m for m in agg_methods if m not in self.ex_agg_methods + self.ex_trans_methods]
        self.df = None

    def fit(self, input_df, y=None):
        new_df = []
        for agg_method in self.agg_methods:
            for col in self.group_values:
                new_col = f"agg_{agg_method}_{col}_grpby_{self.group_key}"
                df_agg = (input_df[[col] + [self.group_key]].groupby(self.group_key)[[col]].agg(agg_method))
                df_agg.columns = [new_col]
                new_df.append(df_agg)
        self.df = pd.concat(new_df, axis=1).reset_index()
        if len(self.ex_agg_methods) != 0:
            self.df = self.ex_aggregation(self.df)

    def transform(self, input_df):
        output_df = pd.merge(input_df[[self.group_key]], self.df, on=self.group_key, how="left")
        if len(self.ex_trans_methods) != 0:
            output_df = self.ex_transform(input_df, output_df)
        output_df.drop(self.group_key, axis=1, inplace=True)
        return output_df

    def ex_aggregation(self, df):
        if "max-min" in self.ex_agg_methods:
            df[self._get_col("max-min")] = df[self._get_col("max")].values - df[self._get_col("min")].values
        if "min-mean" in self.ex_agg_methods:
            df[self._get_col("min-mean")] = df[self._get_col("min")].values - df[self._get_col("mean")].values
        if "max-mean" in self.ex_agg_methods:
            df[self._get_col("max-mean")] = df[self._get_col("max")].values - df[self._get_col("mean")].values
        return df

    def ex_transform(self, df1, df2):
        """
        df1: input_df
        df2: output_df
        return: output_df (added ex transformed features)
        """

        if "val-mean" in self.ex_trans_methods:
            df2[self._get_col("val-mean")] = df1[self.group_values].values - df2[self._get_col("mean")].values
        if "z-score" in self.ex_trans_methods:
            df2[self._get_col("z-score")] = (df1[self.group_values].values - df2[self._get_col("mean")].values) \
                                            / (df2[self._get_col("std")].values + 1e-3)
        return df2

    def _get_col(self, method):
        return np.sort([f"agg_{method}_{group_val}_grpby_{self.group_key}" for group_val in self.group_values])


class PivotingBlock(BaseBlock):
    def __init__(self, idx, col, val, decomposer=PCA(n_components=4), name=""):
        """
        :param idx: index of pivot table
        :param col: columns of pivot table
        :param val: aggregated feature
        :return: DataFrame(columns=col, index=idx)
        """
        self.idx = idx
        self.col = col
        self.val = val
        self.decomposer = decomposer
        self.name = name
        self.df = None

    def fit(self, input_df, y=None):
        _df = input_df.astype(str).pivot_table(
            index=self.idx,
            columns=self.col,
            values=self.val,
            aggfunc='count',
        ).reset_index()

        idx = _df[self.idx]
        _df.drop(self.idx, axis=1, inplace=True)
        _df = _df.div(_df.sum(axis=1), axis=0).fillna(0)

        if self.decomposer is not None:
            self.df = pd.DataFrame(self.decomposer.fit_transform(_df))
            self.df.columns = [f"{i:03}" for i in range(self.df.shape[1])]
        else:
            self.df = _df.copy()

        self.df.columns = [f"pivot_{self.idx}_{self.col}{self.name}:{s}" for s in self.df.columns]
        self.df[self.idx] = idx

    def transform(self, input_df):
        output_df = pd.merge(input_df[[self.idx]], self.df, on=self.idx, how="left").drop(self.idx, axis=1)
        return output_df


class UniqueCatCountingBlock(BaseBlock):
    def __init__(self, group_key, group_values):
        self.group_values = group_values
        self.group_key = group_key

        self.df = None

    def fit(self, input_df, y=None):
        new_df = []
        for col in self.group_values:
            new_col = f"num_cat_{col}_grpby_{self.group_key}"
            df_agg = (input_df[[col] + [self.group_key]]
                      .groupby(self.group_key, as_index=True)
                      .agg({col: pd.Series.nunique}))
            df_agg.columns = [new_col]
            new_df.append(df_agg)
            self.df = pd.concat(new_df, axis=1).reset_index()

    def transform(self, input_df):
        output_df = pd.merge(input_df[[self.group_key]], self.df, on=self.group_key, how="left")
        output_df.drop(self.group_key, axis=1, inplace=True)
        return output_df


class RankingBlock(BaseBlock):
    def __init__(self, group_key, group_values):
        self.group_key = group_key
        self.group_values = group_values

        self.df = None

    def fit(self, input_df, y=None):
        new_df = []
        new_cols = []
        for col in self.group_values:
            new_cols.append(f"ranking_{col}_grpby_{self.group_key}")
            df_agg = (input_df[[col] + [self.group_key]]
                      .groupby(self.group_key)[col].rank(ascending=False, method='min'))
            new_df.append(df_agg)
            self.df = pd.concat(new_df, axis=1)
        self.df.columns = new_cols

    def transform(self, input_df):
        return self.df


class TargetEncodingBlock(BaseBlock):
    def __init__(self, target_col, input_cols, fold=None):
        self.fold = KFold(n_splits=5, shuffle=True) if fold is None else fold
        self.target_col = target_col
        self.input_cols = input_cols

        self.encoder = None

    def fit(self, input_df, y=None):
        self.encoder = xfeat.TargetEncoder(
            input_cols=self.input_cols,
            target_col=self.target_col,
            fold=self.fold,
            output_suffix=""
        )

        _ = self.encoder.fit_transform(input_df)

    def transform(self, input_df):
        output_df = self.encoder.transform(input_df)[self.input_cols]
        output_df.columns = [f"target_encoded_{c}" for c in self.input_cols]
        return output_df
