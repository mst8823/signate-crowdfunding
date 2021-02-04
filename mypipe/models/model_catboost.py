from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from mypipe.models.model_base import BaseModel


class MyCatClassifier(BaseModel):

    def build_model(self):
        model = CatBoostClassifier(**self.params)
        return model

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        self.model = self.build_model()
        tr_pool = Pool(tr_x, tr_y)
        va_pool = Pool(va_x, va_y)
        # fitting - train
        self.model.fit(tr_pool,
                  eval_set=[va_pool],
                  use_best_model=True,
                  verbose=10)

    def predict(self, x):
        preds = self.model.predict(x, prediction_type='Probability')[:, 1]
        return preds

    def _get_feature_importance(self):
        return self.model.feature_importances_


class MyCatRegressor(MyCatClassifier):
    def build_model(self):
        model = CatBoostRegressor(**self.params)
        return model

    def predict(self, x):
        preds = self.model.predict(x)
        return preds
