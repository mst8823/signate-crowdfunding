import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import inspection
from sklearn.metrics import make_scorer
from mypipe.utils import Util, Logger


class Runner:

    def __init__(self, config, run_params, model_params, Model, features):
        """
        :param config: config module
        :param run_params:
                        {"config": config,
                        "metrics": optimized_f1,
                        "fold": make_skf,
                        "select_features": "tree_importance",
                        "feats_select_num": 600,
                        "seeds": [0, 1, 2],
                        "ensemble": None}

        :param model_params: {model parameter dictionary}
        :param Model: Wrapper Model class
        :param features: {"train_x":train_x, "train_y":train_y, "test_x":test_x}
        """
        self.run_params = run_params
        self.model_params = model_params
        self.Model = Model
        self.features = features
        self.config = config
        self.run_name = config.RUN_NAME
        self.folds = config.FOLDS
        self.seeds = run_params["seeds"] if run_params["seeds"] is not None else [2020]
        self.logger = Logger(config.REPORTS)

        self.oof = None
        self.preds = None

    def build_model(self, seed, i_fold):
        model_name = f"{self.config.TRAINED}/{self.run_name}_SEED{seed}_FOLD{i_fold}"
        if "random_state" in self.model_params:  # change seed after exp010
            self.model_params["random_state"] = seed
        model = self.Model(
            name=model_name,
            params=self.model_params
        )
        return model

    def get_score(self, y_true, y_pred):
        if self.run_params["metrics"] is not None:
            score = self.run_params["metrics"](y_true, y_pred)
        else:
            raise NotImplementedError
        return score

    def train_fold(self, seed, i_fold):

        train_x = self.load_x_train()
        train_y = self.load_y_train()
        print(train_x.shape)
        tr_idx, va_idx = self.load_index_fold(seed, i_fold)
        tr_x, tr_y = train_x.values[tr_idx], train_y.values[tr_idx]
        va_x, va_y = train_x.values[va_idx], train_y.values[va_idx]

        model = self.build_model(seed, i_fold)
        model.fit(tr_x, tr_y, va_x, va_y)

        va_pred = model.predict(va_x)
        score = self.get_score(va_y, va_pred)
        self.logger.info(f"{self.run_name} - SEED:{seed}, FOLD:{i_fold} >>> {score:.4f}")

        return model, va_idx, va_pred, score

    def run_train_cv(self):
        self.logger.info(f'{self.run_name} - start training cv')
        train_y = self.load_y_train()  # y true

        preds_seeds = []
        for seed in self.seeds:
            preds = []
            va_idxes = []
            scores = []

            for i_fold in range(self.folds):
                model, va_idx, va_pred, score = self.train_fold(seed, i_fold)
                model.save_model()  # save model

                va_idxes.append(va_idx)
                scores.append(score)
                preds.append(va_pred)

            # sort as default
            va_idxes = np.concatenate(va_idxes)
            order = np.argsort(va_idxes)
            preds = np.concatenate(preds, axis=0)
            preds = preds[order]
            preds_seeds.append(preds)
            score = self.get_score(train_y, preds)
            self.logger.info(f'{self.run_name} - SEED:{seed} - score: {score:.4f}')

        oof = np.mean(preds_seeds, axis=0)
        score = self.get_score(train_y, oof)
        Util.dump(oof, f'{self.config.PREDS}/{self.run_name}-oof.pkl')
        self.logger.info(f'{self.run_name} - end training cv - score: {score:.4f}')
        self.oof = oof

    def run_predict_cv(self) -> None:
        self.logger.info(f'{self.run_name} - start prediction cv')
        test_x = self.load_x_test()

        preds_seeds = []
        for seed in self.seeds:
            preds = []
            for i_fold in range(self.folds):
                self.logger.info(f"{self.run_name} >>> SEED:{seed}, FOLD:{i_fold}")
                model = self.build_model(seed, i_fold)
                model.load_model()
                pred = model.predict(test_x)
                preds.append(pred)
            preds = np.mean(preds, axis=0)
            preds_seeds.append(preds)

        preds = np.mean(preds_seeds, axis=0)
        Util.dump(preds, f'{self.config.PREDS}/{self.run_name}-preds.pkl')
        self.logger.info(f'{self.run_name} - end prediction cv')
        self.preds = preds

    def load_index_fold(self, seed, i_fold):
        train_y = self.load_y_train()
        train_x = self.load_x_train()
        fold = self.run_params["fold"](train_x, train_y, seed)
        return fold[i_fold]

    def load_x_train(self):
        file_path = os.path.join(self.config.COLS, "cols.pkl")
        if os.path.isfile(file_path):
            cols = Util.load(file_path)
        else:
            cols = self.get_features_name()
            Util.dump(cols, file_path)
        num = self.run_params["feats_select_num"]
        num = num if num is not None else len(cols)
        cols = cols[:num]
        return self.features["train_x"][cols]

    def load_y_train(self):
        return self.features["train_y"]

    def load_x_test(self):
        file_path = os.path.join(self.config.COLS, "cols.pkl")
        if os.path.isfile(file_path):
            cols = Util.load(file_path)
        else:
            cols = self.get_features_name()
            Util.dump(cols, file_path)
        num = self.run_params["feats_select_num"]
        num = num if num is not None else len(cols)
        cols = cols[:num]
        return self.features["test_x"][cols]

    def get_features_name(self):
        train_x = self.features["train_x"]
        if self.run_params["select_features"] is None:
            cols = train_x.columns.tolist()

        elif self.run_params["select_features"] == "tree_importance":
            imp_df = self.tree_importance()
            cols = imp_df["column"].to_list()  # get selected col names

        elif self.run_params["select_features"] == "permutation_importance":
            imp_df = self.permutation_importance()
            cols = imp_df["column"].to_list()  # get selected col names

        else:
            raise NotImplementedError

        return cols

    def tree_importance(self):
        """
        get GBDT feature importance
        :return: importance df
        """
        name = "tree_importance"
        train_x = self.features["train_x"]
        train_y = self.features["train_y"]

        feature_importance_df = pd.DataFrame()
        fold_idx = self.run_params["fold"](train_x, train_y)
        for i, (tr_idx, va_idx) in enumerate(fold_idx):
            print(f"fold {i} >>>>>")
            tr_x, va_x = train_x.values[tr_idx], train_x.values[va_idx]
            tr_y, va_y = train_y.values[tr_idx], train_y.values[va_idx]

            model = self.build_model(seed=2021, i_fold=i)
            model.fit(tr_x, tr_y, va_x, va_y)

            _df = pd.DataFrame()
            _df['feature_importance'] = model._get_feature_importance()
            _df['column'] = train_x.columns
            _df['fold'] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

        imp_df = self._get_importance_df(feature_importance_df, name)
        return imp_df

    def permutation_importance(self):
        """
        get permutation importance
        :return: importance df
        """
        name = "permutation_importance"
        get_score = make_scorer(self.run_params["metrics"])
        train_x = self.features["train_x"]
        train_y = self.features["train_y"]
        feature_importance_df = pd.DataFrame()
        fold_idx = self.run_params["fold"](train_x, train_y)
        for i, (tr_idx, va_idx) in enumerate(fold_idx):
            print(f"fold {i} >>>>>")
            tr_x, va_x = train_x.values[tr_idx], train_x.values[va_idx]
            tr_y, va_y = train_y.values[tr_idx], train_y.values[va_idx]

            model = self.build_model(seed=2021, i_fold=i)
            model.fit(tr_x, tr_y, va_x, va_y)

            _df = pd.DataFrame()
            result = inspection.permutation_importance(estimator=model.model,
                                                       X=va_x, y=va_y,
                                                       scoring=get_score,
                                                       n_repeats=5,
                                                       n_jobs=-1,
                                                       random_state=2021)
            _df['feature_importance'] = result["importances_mean"]
            _df['column'] = train_x.columns
            _df['fold'] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

        imp_df = self._get_importance_df(feature_importance_df, name)
        return imp_df

    def _get_importance_df(self, feature_importance_df, name):
        order = feature_importance_df.groupby('column').sum()[['feature_importance']]
        order = order.sort_values('feature_importance', ascending=False).index[:50]
        fig, ax = plt.subplots(figsize=(12, max(4, len(order) * .2)))
        sns.boxenplot(data=feature_importance_df,
                      y='column',
                      x='feature_importance',
                      order=order,
                      ax=ax,
                      palette='viridis')
        fig.tight_layout()
        ax.grid()
        ax.set_title(name)
        fig.tight_layout()
        plt.show()

        imp_df = feature_importance_df.groupby("column", as_index=False).mean()
        imp_df = imp_df.sort_values("feature_importance", ascending=False)
        imp_df = imp_df.query('feature_importance > 0')[["column", "feature_importance"]]  # remove importance = 0

        fig.savefig(os.path.join(self.config.REPORTS, f'{self.run_name}_{name}_fig.png'), dpi=120)  # save figure
        imp_df.to_csv(os.path.join(self.config.REPORTS, f'{self.run_name}_{name}_df.csv'), index=False)  # save df
        return imp_df
