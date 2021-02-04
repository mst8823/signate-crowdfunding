import os
from mypipe.utils import Util


class BaseBlock(object):
    def fit(self, input_df, y=None):
        raise NotImplementedError

    def transform(self, input_df):
        raise NotImplementedError

    def fit_transform(self, input_df):
        self.fit(input_df)
        return self.transform(input_df)


# decorator
def feature_decorator(feature_name):
    def _feature(func):
        def wrapper(input_df):
            if os.path.isfile(feature_name):
                output_df = Util.load(feature_name)

            else:
                output_df = func(input_df=input_df)
                Util.dump(output_df, feature_name)
            return output_df

        return wrapper

    return _feature
