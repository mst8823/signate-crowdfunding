from sklearn.linear_model import LogisticRegression
from mypipe.models.model_base import BaseModel


class MyLogisticRegModel(BaseModel):

    def build_model(self):
        model = LogisticRegression(**self.params)
        return model

    def predict(self, x):
        preds = self.model.predict_proba(x)[:, 1]
        return preds
