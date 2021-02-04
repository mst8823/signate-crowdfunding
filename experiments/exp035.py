import os
import re
import warnings

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import texthero as hero

import sys
sys.path.append('../')

from mypipe.config import Config
from mypipe import exp_env
from mypipe.utils import reduce_mem_usage, Logger
from mypipe.runner import Runner
from mypipe.features.base import feature_decorator
from mypipe.features.feat_engine import OneHotEncodingBlock, CountEncodingBlock, GroupingBlock, PivotingBlock, \
    RankingBlock
from mypipe.features.text import BasicCountLangBlock, BasicHTMLTransformerBlock, BasicTextFeatureTransformerBlock, \
    Doc2VecFeatureTransformer, TextVectorizer
from mypipe.models.model_utils import threshold_optimization, visualize_confusion_matrix
from mypipe.models.model_lgbm import MyLGBMModel

# ---------------------------------------------------------------------- #
# TODO: final sum model
warnings.filterwarnings("ignore")
RUN_NAME = "exp035"
config = Config(RUN_NAME, folds=10)
exp_env.make_env(config)
ONLY_MAKE_ENV = False


# ---------------------------------------------------------------------- #

@feature_decorator(os.path.join(config.FEATURE, "goal.pkl"))
def get_goal_features(input_df):
    tmp = input_df["goal"]
    tmp = tmp.replace("100000+", "100000-100000")
    tmp = np.array([g.split("-") for g in tmp], dtype="int")
    output_df = pd.DataFrame(tmp, columns=["goal_min", "goal_max"])
    output_df["goal_upper_flag"] = output_df["goal_min"] == 100000
    output_df["goal_lower_flag"] = output_df["goal_min"] == 1
    output_df["goal_mean"] = output_df[["goal_min", "goal_max"]].mean(axis=1)
    output_df["goal_q25"] = output_df[["goal_min", "goal_max"]].quantile(q=0.25, axis=1)
    output_df["goal_q75"] = output_df[["goal_min", "goal_max"]].quantile(q=0.75, axis=1)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "bins.pkl"))
def get_bins_feature(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
    ], axis=1)
    output_df = pd.DataFrame()
    output_df["bins_duration"] = pd.cut(_input_df["duration"],
                                        bins=[-1, 30, 45, 60, 100],
                                        labels=['bins_d1', 'bins_d2', 'bins_d3', 'bins_d4'])
    output_df["bins_goal"] = pd.cut(_input_df["goal_max"],
                                    bins=[-1, 19999, 49999, 79999, 99999, np.inf],
                                    labels=['bins_g1', 'bins_g2', 'bins_g3', 'bins_g4', 'bins_g5'])
    output_df = output_df.astype(str)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ce.pkl"))
def get_ce_features(input_df):
    _input_df = pd.concat([
        input_df,
        get_cross_cat_features(input_df),
        get_bins_feature(input_df)],
        axis=1)
    cols = [
        "category1",
        "category2",
        "category3",
        "country",
        "country+category1",
        "country+category2",
        "country+category3",
        "bins_duration",
        "bins_goal",
        "bins_DurationGoal",
        "bins_duration+category1",
        "bins_duration+category2",
        "bins_duration+category3",
        "bins_goal+category1",
        "bins_goal+category2",
        "bins_goal+category3",
        "bins_DurationGoal+category1",
        "bins_DurationGoal+category2",
        "bins_DurationGoal+category3",
    ]
    encoder = CountEncodingBlock(cols)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ohe.pkl"))
def get_ohe_features(input_df):
    _input_df = pd.concat([
        input_df,
        get_cross_cat_features(input_df)
    ], axis=1)
    cols = ["country"]
    encoder = OneHotEncodingBlock(cols)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "cross_categorical.pkl"))
def get_cross_cat_features(input_df):
    _input_df = pd.concat([
        input_df,
        get_bins_feature(input_df)
    ], axis=1).astype(str)

    output_df = pd.DataFrame()
    output_df["category3"] = _input_df["category1"] + _input_df["category2"]
    output_df["country+category1"] = _input_df["country"] + _input_df["category1"]
    output_df["country+category2"] = _input_df["country"] + _input_df["category2"]
    output_df["country+category3"] = _input_df["country"] + output_df["category3"]
    output_df["bins_DurationGoal"] = _input_df["bins_duration"] + _input_df["bins_goal"]
    output_df["bins_duration+category1"] = _input_df["bins_duration"] + _input_df["category1"]
    output_df["bins_duration+category2"] = _input_df["bins_duration"] + _input_df["category2"]
    output_df["bins_duration+category3"] = _input_df["bins_duration"] + output_df["category3"]
    output_df["bins_goal+category1"] = _input_df["bins_goal"] + _input_df["category1"]
    output_df["bins_goal+category2"] = _input_df["bins_goal"] + _input_df["category2"]
    output_df["bins_goal+category3"] = _input_df["bins_goal"] + output_df["category3"]
    output_df["bins_DurationGoal+category1"] = output_df["bins_DurationGoal"] + _input_df["category1"]
    output_df["bins_DurationGoal+category2"] = output_df["bins_DurationGoal"] + _input_df["category2"]
    output_df["bins_DurationGoal+category3"] = output_df["bins_DurationGoal"] + output_df["category3"]
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "cross_numerical.pkl"))
def get_cross_num_features(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_basic_html_features(input_df)
    ], axis=1)

    output_df = pd.DataFrame()
    output_df["ratio_goalMax_duration"] = _input_df["goal_max"] / (_input_df["duration"] + 1)
    output_df["ratio_goalMin_duration"] = _input_df["goal_min"] / (_input_df["duration"] + 1)
    output_df["ratio_goalMean_duration"] = _input_df["goal_mean"] / (_input_df["duration"] + 1)
    output_df["prod_goalMax_duration"] = _input_df["goal_max"] * (_input_df["duration"])
    output_df["prod_goalMin_duration"] = _input_df["goal_min"] * (_input_df["duration"])
    output_df["prod_goalMean_duration"] = _input_df["goal_mean"] * (_input_df["duration"])
    output_df["html_num_figs_and_imgs"] = _input_df["html__num_imgs"] + _input_df["html__num_figs"]
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_country.pkl"))
def agg_country(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)
    group_key = "country"
    group_values = [
        "goal_min",
        "goal_max",
        "goal_mean",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
    ]
    agg_methods = ["min", "max", "mean", "std", "count", "max-min", "z-score"]
    encoder = GroupingBlock(group_key, group_values, agg_methods)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_category1.pkl"))
def agg_category1(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_cat_features(input_df),
        get_cross_num_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df),
    ], axis=1)
    group_key = "category1"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",

    ]
    agg_methods = ["min", "max", "mean", "std", "count", "max-min", "z-score"]
    encoder = GroupingBlock(group_key, group_values, agg_methods)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_category2.pkl"))
def agg_category2(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_cat_features(input_df),
        get_cross_num_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df),
    ], axis=1)
    group_key = "category2"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    agg_methods = ["min", "max", "mean", "std", "count", "max-min", "z-score"]
    encoder = GroupingBlock(group_key, group_values, agg_methods)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_category3.pkl"))
def agg_category3(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_cat_features(input_df),
        get_cross_num_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df),
    ], axis=1)
    group_key = "category3"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    agg_methods = ["min", "max", "mean", "std", "count", "max-min", "z-score"]
    encoder = GroupingBlock(group_key, group_values, agg_methods)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_country+category1.pkl"))
def agg_country_category1(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_cat_features(input_df),
        get_cross_num_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df),
    ], axis=1)
    group_key = "country+category1"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    agg_methods = ["min", "max", "mean", "std", "count", "max-min", "z-score"]
    encoder = GroupingBlock(group_key, group_values, agg_methods)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_country+category2.pkl"))
def agg_country_category2(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_cat_features(input_df),
        get_cross_num_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df),
    ], axis=1)
    group_key = "country+category2"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    agg_methods = ["min", "max", "mean", "std", "count", "max-min", "z-score"]
    encoder = GroupingBlock(group_key, group_values, agg_methods)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_country+category3.pkl"))
def agg_country_category3(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_cat_features(input_df),
        get_cross_num_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df),
    ], axis=1)
    group_key = "country+category3"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    agg_methods = ["min", "max", "mean", "std", "count", "max-min", "z-score"]
    encoder = GroupingBlock(group_key, group_values, agg_methods)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_bins_DurationGoal.pkl"))
def agg_bins_duration_goal(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df),

    ], axis=1)
    group_key = "bins_DurationGoal"
    group_values = [
        "duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
    ]
    agg_methods = ["min", "max", "mean", "std", "count", "max-min", "z-score"]
    encoder = GroupingBlock(group_key, group_values, agg_methods)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_bins_goal+category3.pkl"))
def agg_bins_goal_category3(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df),

    ], axis=1)
    group_key = "bins_goal+category3"
    group_values = [
        "duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
    ]
    agg_methods = ["min", "max", "mean", "std", "count", "max-min", "z-score"]
    encoder = GroupingBlock(group_key, group_values, agg_methods)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_bins_goal+category2.pkl"))
def agg_bins_goal_category2(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df),

    ], axis=1)
    group_key = "bins_goal+category2"
    group_values = [
        "duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
    ]
    agg_methods = ["min", "max", "mean", "std", "count", "max-min", "z-score"]
    encoder = GroupingBlock(group_key, group_values, agg_methods)
    output_df = encoder.fit_transform(_input_df)
    return output_df


# ------------------------------------------------------------------------------- #
@feature_decorator(os.path.join(config.FEATURE, "ranking_country.pkl"))
def rank_country(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)
    group_key = "country"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key=group_key, group_values=group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ranking_category1.pkl"))
def rank_category1(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)

    group_key = "category1"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key=group_key, group_values=group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ranking_category2.pkl"))
def rank_category2(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)

    group_key = "category2"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key=group_key, group_values=group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ranking_category3.pkl"))
def rank_category3(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df),

    ], axis=1)
    group_key = "category3"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key, group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ranking_country_category1.pkl"))
def rank_country_category1(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)

    group_key = "country+category1"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key=group_key, group_values=group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ranking_country_category2.pkl"))
def rank_country_category2(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)

    group_key = "country+category2"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key=group_key, group_values=group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ranking_country_category3.pkl"))
def rank_country_category3(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)

    group_key = "country+category3"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key=group_key, group_values=group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ranking_bins_DurationGoal.pkl"))
def rank_bins_duration_goal(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)

    group_key = "bins_DurationGoal"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key=group_key, group_values=group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ranking_bins_goal+category1.pkl"))
def rank_bins_goal_category1(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)

    group_key = "bins_goal+category1"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key=group_key, group_values=group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ranking_bins_goal+category2.pkl"))
def rank_bins_goal_category2(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)

    group_key = "bins_goal+category2"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key=group_key, group_values=group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "ranking_bins_goal+category3.pkl"))
def rank_bins_goal_category3(input_df):
    _input_df = pd.concat([
        input_df,
        get_goal_features(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
        get_basic_text_features__raw(input_df),
        get_basic_html_features(input_df)
    ], axis=1)

    group_key = "bins_goal+category3"
    group_values = [
        "goal_min",
        "goal_max",
        "duration",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "html_content_num_punctuation",
        "html_content_num_chars",
        "html_content_num_exclamation_marks",
        "html_content_num_unique_words",
        "html__num_links",
        "html__num_figs",
        "html__num_as",
        "html__num_ps",
        "html__num_divs",
    ]
    encoder = RankingBlock(group_key=group_key, group_values=group_values)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "agg_ranking.pkl"))
def agg_ranking(input_df):
    _input_df = pd.concat([
        input_df,
        rank_category1(input_df),
        rank_category2(input_df),
        rank_category3(input_df),
        rank_bins_goal_category1(input_df),
        rank_bins_goal_category2(input_df),
        rank_bins_goal_category3(input_df),
        get_cross_num_features(input_df)
    ], axis=1)
    group_keys = [
        "ranking_prod_goalMin_duration_grpby_category1",
        "ranking_prod_goalMax_duration_grpby_category1",
        "ranking_prod_goalMin_duration_grpby_category2",
        "ranking_prod_goalMax_duration_grpby_category2",
        "ranking_prod_goalMin_duration_grpby_category3",
        "ranking_prod_goalMax_duration_grpby_category3",
        "ranking_prod_goalMin_duration_grpby_bins_goal+category1",
        "ranking_prod_goalMax_duration_grpby_bins_goal+category1",
        "ranking_prod_goalMin_duration_grpby_bins_goal+category2",
        "ranking_prod_goalMax_duration_grpby_bins_goal+category2",
        "ranking_prod_goalMin_duration_grpby_bins_goal+category3",
        "ranking_prod_goalMax_duration_grpby_bins_goal+category3",
    ]
    group_values = [
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
    ]
    agg_methods = ["min", "max", "std", "mean", "z-score"]
    _input_df[group_keys] = _input_df[group_keys].astype(str)
    output_df = []
    for group_key in group_keys:
        encoder = GroupingBlock(group_key=group_key, group_values=group_values, agg_methods=agg_methods)
        _df = encoder.fit_transform(_input_df)
        output_df.append(_df)
    return pd.concat(output_df, axis=1)


# ------------------------------------------------------------------------------- #

@feature_decorator(os.path.join(config.FEATURE, "pivot_cat1_cat2__pca8.pkl"))
def pivot_cat1_cat2__pca8(input_df):
    idx = "category1"
    col = "category2"
    val = "duration"
    decomposer = PCA(n_components=8, random_state=2021)
    name = "_pca"
    encoder = PivotingBlock(idx, col, val, decomposer, name)
    output_df = encoder.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "pivot_cat3_cat1__pca8.pkl"))
def pivot_cat3_cat1__pca8(input_df):
    _input_df = pd.concat([
        input_df,
        get_cross_cat_features(input_df),
    ], axis=1)
    idx = "category3"
    col = "category1"
    val = "duration"
    decomposer = PCA(n_components=8, random_state=2021)
    name = "_pca"
    encoder = PivotingBlock(idx, col, val, decomposer, name)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "pivot_cat3_cat2__pca8.pkl"))
def pivot_cat1_cat2__pca8(input_df):
    _input_df = pd.concat([
        input_df,
        get_cross_cat_features(input_df),
    ], axis=1)
    idx = "category3"
    col = "category2"
    val = "duration"
    decomposer = PCA(n_components=8, random_state=2021)
    name = "_pca"
    encoder = PivotingBlock(idx, col, val, decomposer, name)
    output_df = encoder.fit_transform(_input_df)
    return output_df


# ------------------------------------------------------------------------------- #
def cleansing_hero_remove_html_tags(input_df, text_col):
    # only remove html tags, do not remove punctuation
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_stopwords,
        hero.preprocessing.remove_whitespace,
        hero.preprocessing.stem
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts


def cleansing_hero_only_text(input_df, text_col):
    # get only text (remove html tags, punctuation & digits)
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        hero.preprocessing.remove_stopwords,
        hero.preprocessing.remove_whitespace,
        hero.preprocessing.stem
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts


def get_html_tags_only(input_df, text_col):
    htmls = input_df[text_col]
    html_tags = []
    for html in htmls:
        tmp = re.sub(r"\s", "", html)
        tmp = re.sub(r"\d", "", tmp)
        tmp = " ".join(re.findall(r"(?<=<).*?(?=>)", tmp))
        html_tags.append(tmp)

    return pd.Series(html_tags)


@feature_decorator(os.path.join(config.FEATURE, "basic_text_features__raw.pkl"))
def get_basic_text_features__raw(input_df):
    bte = BasicTextFeatureTransformerBlock(["html_content"], cleansing_hero=None)
    output_df = bte.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "basic_text_features__removed_html_tags.pkl"))
def get_basic_text_features__removed_html_tags(input_df):
    bte = BasicTextFeatureTransformerBlock(["html_content"],
                                           cleansing_hero=cleansing_hero_remove_html_tags,
                                           name="removed_html_tags")
    output_df = bte.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "basic_html_features.pkl"))
def get_basic_html_features(input_df):
    base_html = BasicHTMLTransformerBlock(["html_content"])
    output_df = base_html.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "basic_count_lang_features.pkl"))
def get_basic_count_lang_features(input_df):
    bcl = BasicCountLangBlock(["html_content"])
    output_df = bcl.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "doc2vec_features__only_text.pkl"))
def get_doc2vec_features__only_text(input_df):
    d2v = Doc2VecFeatureTransformer(["html_content"],
                                    cleansing_hero=cleansing_hero_only_text,
                                    params={"vector_size": 128, "min_count": 1, "epochs": 30, "seed": 2021}
                                    )
    output_df = d2v.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "doc2vec_features__removed_html_tags.pkl"))
def get_doc2vec_features__removed_html_tags(input_df):
    d2v = Doc2VecFeatureTransformer(["html_content"],
                                    cleansing_hero=cleansing_hero_remove_html_tags,
                                    params={"vector_size": 128, "min_count": 1, "epochs": 30, "seed": 2021},
                                    name="doc2vec_removed_html_tags"
                                    )
    output_df = d2v.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "text_vector__html_tags_only__count_sdv256.pkl"))
def get_text_vector__html_tags_only__count_sdv256(input_df):
    tv = TextVectorizer(["html_content"],
                        cleansing_hero=get_html_tags_only,
                        vectorizer=CountVectorizer(),
                        transformer=TruncatedSVD(n_components=256, random_state=2021),
                        name="html_tags_only__count_sdv"
                        )
    output_df = tv.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "text_vector__html_tags_only_chtoken__count_sdv256.pkl"))
def get_text_vector__html_tags_only_chtoken__count_sdv256(input_df):
    tv = TextVectorizer(["html_content"],
                        cleansing_hero=get_html_tags_only,
                        vectorizer=CountVectorizer(token_pattern=r"\S+"),
                        transformer=TruncatedSVD(n_components=256, random_state=2021),
                        name="html_tags_only_chtoken__count_sdv"
                        )
    output_df = tv.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "text_vector__raw__count_sdv128.pkl"))
def get_text_vector__raw__count_sdv128(input_df):
    tv = TextVectorizer(["html_content"],
                        cleansing_hero=None,
                        vectorizer=CountVectorizer(),
                        transformer=TruncatedSVD(n_components=128, random_state=2021),
                        name="html_content__raw__count_sdv"
                        )
    output_df = tv.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "text_vector__removed_html_tags__count_sdv128.pkl"))
def get_text_vector__removed_html_tags__count_sdv128(input_df):
    tv = TextVectorizer(["html_content"],
                        cleansing_hero=cleansing_hero_remove_html_tags,
                        vectorizer=CountVectorizer(),
                        transformer=TruncatedSVD(n_components=128, random_state=2021),
                        name="html_content__removed_html_tags__count_sdv"
                        )
    output_df = tv.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "text_vector__only_text__count_sdv128.pkl"))
def get_text_vector__only_text__count_sdv128(input_df):
    tv = TextVectorizer(["html_content"],
                        vectorizer=CountVectorizer(),
                        cleansing_hero=cleansing_hero_only_text,
                        transformer=TruncatedSVD(n_components=128, random_state=2021),
                        name="html_content__only_text__count_sdv"
                        )
    output_df = tv.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "text_vector__html_tags_only__tfidf_sdv256.pkl"))
def get_text_vector__html_tags_only__tfidf_sdv256(input_df):
    tv = TextVectorizer(["html_content"],
                        cleansing_hero=get_html_tags_only,
                        vectorizer=TfidfVectorizer(),
                        transformer=TruncatedSVD(n_components=256, random_state=2021),
                        name="html_tags_only__tfidf_sdv"
                        )
    output_df = tv.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "text_vector__html_tags_only_chtoken__tfidf_sdv256.pkl"))
def get_text_vector__html_tags_only_chtoken__tfidf_sdv256(input_df):
    tv = TextVectorizer(["html_content"],
                        cleansing_hero=get_html_tags_only,
                        vectorizer=TfidfVectorizer(token_pattern=r"\S+"),
                        transformer=TruncatedSVD(n_components=256, random_state=2021),
                        name="html_tags_only_chtoken__tfidf_sdv"
                        )
    output_df = tv.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "text_vector__raw__tfidf_sdv128.pkl"))
def get_text_vector__raw__tfidf_sdv128(input_df):
    tv = TextVectorizer(["html_content"],
                        cleansing_hero=None,
                        vectorizer=TfidfVectorizer(),
                        transformer=TruncatedSVD(n_components=128, random_state=2021),
                        name="html_content__raw___tfidf_sdv"
                        )
    output_df = tv.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "text_vector__removed_html_tags__tfidf_sdv128.pkl"))
def get_text_vector__removed_html_tags__tfidf_sdv128(input_df):
    tv = TextVectorizer(["html_content"],
                        cleansing_hero=cleansing_hero_remove_html_tags,
                        vectorizer=TfidfVectorizer(),
                        transformer=TruncatedSVD(n_components=128, random_state=2021),
                        name="html_content__removed_html_tags__tfidf_sdv"
                        )
    output_df = tv.fit_transform(input_df)
    return output_df


@feature_decorator(os.path.join(config.FEATURE, "text_vector__only_text__tfidf_sdv128.pkl"))
def get_text_vector__only_text__tfidf_sdv128(input_df):
    tv = TextVectorizer(["html_content"],
                        vectorizer=TfidfVectorizer(),
                        cleansing_hero=cleansing_hero_only_text,
                        transformer=TruncatedSVD(n_components=128, random_state=2021),
                        name="html_content__only_text__tfidf_sdv"

                        )
    output_df = tv.fit_transform(input_df)
    return output_df


# ------------------------------------------------------------------------------- #

# main func for preprocessing
def preprocess():
    # load data
    train = pd.read_csv(os.path.join(config.INPUT, "train.csv"))
    test = pd.read_csv(os.path.join(config.INPUT, "test.csv"))
    input_df = pd.concat([train, test]).reset_index(drop=True)  # use concat data

    # load process functions
    process_blocks = get_process_blocks()

    # preprocess
    output_df = []
    for func in tqdm(process_blocks):
        _df = func(input_df)
        output_df.append(reduce_mem_usage(_df, False))  # reduce mem
    output_df = pd.concat(output_df, axis=1)

    # separate train and test
    train_x = output_df.iloc[:len(train)]
    test_x = output_df.iloc[len(train):].reset_index(drop=True)
    train_y = train[config.TARGET]

    feats = {"train_x": train_x, "train_y": train_y, "test_x": test_x}
    return feats


def get_process_blocks():
    process_blocks = [
        get_goal_features,
        get_ce_features,
        get_ohe_features,
        # get_cross_cat_features,
        get_cross_num_features,
        agg_country,
        agg_category1,
        agg_category2,
        agg_category3,
        agg_country_category1,
        agg_country_category2,
        agg_country_category3,
        agg_bins_duration_goal,
        agg_bins_goal_category3,
        agg_bins_goal_category2,
        rank_country,
        rank_category1,
        rank_category2,
        rank_category3,
        rank_country_category1,
        rank_country_category2,
        rank_country_category3,
        rank_bins_duration_goal,
        rank_bins_goal_category1,
        rank_bins_goal_category2,
        rank_bins_goal_category3,
        agg_ranking,
        pivot_cat1_cat2__pca8,
        pivot_cat1_cat2__pca8,
        pivot_cat3_cat1__pca8,
        get_basic_text_features__raw,
        get_basic_text_features__removed_html_tags,
        get_basic_html_features,
        get_basic_count_lang_features,
        get_doc2vec_features__only_text,
        get_doc2vec_features__removed_html_tags,
        get_text_vector__html_tags_only__count_sdv256,
        get_text_vector__html_tags_only_chtoken__count_sdv256,
        get_text_vector__raw__count_sdv128,
        get_text_vector__removed_html_tags__count_sdv128,
        get_text_vector__only_text__count_sdv128,
        get_text_vector__html_tags_only__tfidf_sdv256,
        get_text_vector__html_tags_only_chtoken__tfidf_sdv256,
        get_text_vector__raw__tfidf_sdv128,
        get_text_vector__removed_html_tags__tfidf_sdv128,
        get_text_vector__only_text__tfidf_sdv128
    ]
    return process_blocks


# ------------------------------------------------------------------------------- #
# ------ Train & predict ---------


# get train
def get_train_data():
    all_features = preprocess()
    x = all_features["train_x"]
    y = all_features["train_y"]
    return x, y


# get test
def get_test_data():
    all_features = preprocess()
    x = all_features["test_x"]
    return x


# define metrics
def optimized_f1(y_true, y_pred):
    bt = threshold_optimization(y_true, y_pred, metrics=f1_score)
    score = f1_score(y_true, y_pred >= bt)
    return score


# make fold
def make_skf(train_x, train_y, random_state=2020):
    skf = StratifiedKFold(n_splits=config.FOLDS, shuffle=True, random_state=random_state)
    return list(skf.split(train_x, train_y))


# visualize result
def visualize_result(y_true, y_pred):
    bt = threshold_optimization(y_true, y_pred, optimized_f1)
    fig = visualize_confusion_matrix(y_true, y_pred >= bt)
    fig.savefig(os.path.join(config.REPORTS, f'{config.RUN_NAME}'), dpi=120)  # save figure


# get label
def get_label(target, oof, preds):
    bt = threshold_optimization(target, oof, optimized_f1)
    print(f"Best Threshold is {bt}")
    labels = preds >= bt
    return np.array(labels, dtype="int32")


# create submission
def create_submission(preds):
    sample_sub = pd.read_csv(os.path.join(config.INPUT, "sample_submit.csv"), header=None)
    sample_sub[1] = preds
    sample_sub.to_csv(os.path.join(config.SUBMISSION, f'{config.RUN_NAME}_new.csv'), index=False, header=False)


def main():
    logger = Logger(config.REPORTS)

    # preprocess
    train_x, train_y = get_train_data()
    test_x = get_test_data()

    # set run params
    run_params = {
        "config": config,
        "metrics": optimized_f1,
        "fold": make_skf,
        "select_features": "tree_importance",
        "feats_select_num": 600,
        "seeds": [0, 1, 2, 3, 4, 5, 6],
        "ensemble": None
    }

    # set model params
    model_params = {
        "n_estimators": 10000,
        "objective": 'binary',
        "learning_rate": 0.01,
        "num_leaves": 31,
        "random_state": 2021,
        "n_jobs": -1,
        "importance_type": "gain",
        'colsample_bytree': .5,
        "reg_lambda": 5,
    }

    # save log
    logger.info(f"RUN NAME : {config.RUN_NAME}")
    logger.info(f"run params : {run_params}")
    logger.info(f"model params : {model_params}")
    logger.info(f"folds: {config.FOLDS}")

    features = {
        "train_x": train_x,
        "test_x": test_x,
        "train_y": train_y
    }
    print(train_x.shape)
    print(test_x.shape)

    # run
    runner = Runner(config=config,
                    run_params=run_params,
                    model_params=model_params,
                    Model=MyLGBMModel,
                    features=features)
    runner.run_train_cv()
    runner.run_predict_cv()

    # plot result
    visualize_result(train_y, runner.oof)

    # make submission
    create_submission(preds=get_label(train_y, runner.oof, runner.preds))


# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not ONLY_MAKE_ENV:
        main()
