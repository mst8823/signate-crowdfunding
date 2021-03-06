from xgboost import XGBModel
from mypipe.models.model_base import BaseModel


class MyXGBModel(BaseModel):

    def build_model(self):
        model = XGBModel(**self.params)
        return model

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        self.model = self.build_model()
        self.model.fit(tr_x, tr_y,
                       eval_set=[(va_x, va_y)],
                       early_stopping_rounds=100,
                       verbose=True)

    def _get_feature_importance(self):
        return self.model.feature_importances_
