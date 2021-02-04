import re
import pandas as pd

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from bs4 import BeautifulSoup
from mypipe.features.base import BaseBlock


# ------------------------------------------------------------------- #

class BasicTextFeatureTransformerBlock(BaseBlock):
    def __init__(self, text_columns, cleansing_hero=None, name=""):
        """

        :param text_columns:

            ["col1", "col2", ...,]

        :param cleansing_hero:

            def cleansing_hero(input_df, text_col):
                custom_pipeline = [
                    preprocessing.fillna,
                    preprocessing.remove_urls,
                    preprocessing.remove_html_tags,
                    preprocessing.lowercase,
                    preprocessing.remove_digits,
                    preprocessing.remove_punctuation,
                    preprocessing.remove_diacritics,
                    preprocessing.remove_stopwords,
                    preprocessing.remove_whitespace,
                    preprocessing.stem
                ]
                texts = hero.clean(input_df[text_col], custom_pipeline)
                return texts

        :param name:
        """
        self.text_columns = text_columns
        self.cleansing_hero = cleansing_hero
        self.name = name

        self.df = None

    def fit(self, input_df, y=None):
        output_df = pd.DataFrame()
        output_df[self.text_columns] = input_df[self.text_columns].astype(str).fillna('missing')
        for c in self.text_columns:
            if self.cleansing_hero is not None:
                output_df[c] = self.cleansing_hero(output_df, c)
            output_df = self._get_features(output_df, c)

        self.df = output_df

    def transform(self, input_df):
        return self.df

    def _get_features(self, dataframe, column):
        output_df = pd.DataFrame()
        output_df[column + self.name + '_num_chars'] = dataframe[column].apply(len)
        output_df[column + self.name + '_num_exclamation_marks'] = dataframe[column].apply(lambda x: x.count('!'))
        output_df[column + self.name + '_num_question_marks'] = dataframe[column].apply(lambda x: x.count('?'))
        output_df[column + self.name + '_num_tag_marks'] = dataframe[column].apply(lambda x: x.count('<'))

        output_df[column + self.name + '_num_punctuation'] = dataframe[column].apply(
            lambda x: sum(x.count(w) for w in '.,;:'))
        output_df[column + self.name + '_num_symbols'] = dataframe[column].apply(
            lambda x: sum(x.count(w) for w in '*&$%'))
        output_df[column + self.name + '_num_words'] = dataframe[column].apply(lambda x: len(x.split()))
        output_df[column + self.name + '_num_unique_words'] = dataframe[column].apply(
            lambda x: len(set(w for w in x.split())))
        output_df[column + self.name + '_num_smiles'] = dataframe[column].apply(
            lambda x: sum(x.count(w) for w in (':-)', ':)', ';-)', ';)')))

        output_df[column + self.name + '_words_vs_unique'] = \
            output_df[column + self.name + '_num_unique_words'] / output_df[column + self.name + '_num_words']

        output_df[column + self.name + '_words_vs_chars'] = \
            output_df[column + self.name + '_num_words'] / output_df[column + self.name + '_num_chars']
        return output_df


class BasicCountLangBlock(BaseBlock):
    def __init__(self,
                 text_columns,
                 cleansing_hero=None,
                 name="_lang"
                 ):
        self.text_columns = text_columns
        self.cleansing_hero = cleansing_hero
        self.name = name
        self.df = None

    def fit(self, input_df, y=None):
        output_df = pd.DataFrame()
        output_df[self.text_columns] = input_df[self.text_columns].astype(str).fillna('missing')
        for c in self.text_columns:
            if self.cleansing_hero is not None:
                output_df[c] = self.cleansing_hero(output_df, c)

            output_df = self._get_features(output_df, c)
        self.df = output_df

    def transform(self, input_df):
        return self.df

    def _get_features(self, dataframe, column):
        output_df = pd.DataFrame()
        output_df[column + self.name + '_num_chars'] = dataframe[column].apply(len)
        output_df[column + self.name + '_num_words'] = dataframe[column].apply(lambda x: len(x.split()))
        output_df[column + self.name + '_num_unique_words'] = dataframe[column].apply(
            lambda x: len(set(w for w in x.split())))
        output_df[column + self.name + '_num_en_chars'] = dataframe[column].apply(lambda x: self._count_roma_word(x))
        output_df[column + self.name + '_num_ja_chars'] = dataframe[column].apply(
            lambda x: self._count_japanese_word(x))
        output_df[column + self.name + '_num_ja_chars_vs_chars'] \
            = output_df[column + self.name + '_num_ja_chars'] / (output_df[column + self.name + '_num_chars'] + 1)
        output_df[column + self.name + '_num_en_chars_vs_chars'] \
            = output_df[column + self.name + '_num_en_chars'] / (output_df[column + self.name + '_num_chars'] + 1)

        return output_df

    def _count_japanese_word(self, strings):
        p = re.compile(
            "[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]|[\uD840-\uD87F]|[\uDC00-\uDFFF]|[ぁ-んァ-ヶ]|[Ａ-Ｚ]|[ｦ-ﾟ]|[ａ-ｚ]")
        count_ja_words = len(p.findall(strings))
        return count_ja_words

    def _count_roma_word(self, strings):
        count_en_word = len(re.findall("[a-zA-Z]", strings))
        return count_en_word


class BasicHTMLTransformerBlock(BaseBlock):
    def __init__(self, text_columns, name=""):
        self.text_columns = text_columns
        self.name = name
        self.df = None

    def fit(self, input_df, y=None):
        output_df = pd.DataFrame()
        output_df[self.text_columns] = input_df[self.text_columns].astype(str).fillna('missing')
        for c in self.text_columns:
            output_df[c] = [BeautifulSoup(html, 'html.parser') for html in output_df[c]]
            output_df = self._get_features(output_df, c)
        self.df = output_df

    def transform(self, input_df):
        return self.df

    def _get_features(self, dataframe, column):
        output_df = pd.DataFrame()
        soups = dataframe[column]
        output_df[f"html_{self.name}_num_links"] = [len([url.get('href') for url in soup.find_all('a')]) for soup in
                                                    soups]
        output_df[f"html_{self.name}_num_divs"] = [len(soup.find_all('div')) for soup in soups]
        output_df[f"html_{self.name}_num_figs"] = [len(soup.find_all('figure')) for soup in soups]
        output_df[f"html_{self.name}_num_ps"] = [len(soup.find_all('p')) for soup in soups]
        output_df[f"html_{self.name}_num_as"] = [len(soup.find_all('a')) for soup in soups]
        output_df[f"html_{self.name}_num_brs"] = [len(soup.find_all('br')) for soup in soups]
        output_df[f"html_{self.name}_num_h1s"] = [len(soup.find_all('h1')) for soup in soups]
        output_df[f"html_{self.name}_num_figcaps"] = [len(soup.find_all('figcaption')) for soup in soups]
        output_df[f"html_{self.name}_num_lis"] = [len(soup.find_all('li')) for soup in soups]
        output_df[f"html_{self.name}_num_imgs"] = [len(soup.find_all('img')) for soup in soups]
        output_df[f"html_{self.name}_num_ALL"] = output_df.sum(axis=1)
        output_df[f"html_{self.name}_num_MEAN"] = output_df.mean(axis=1)
        output_df[f"html_{self.name}_num_STD"] = output_df.std(axis=1)  
        return output_df


class TextVectorizer(BaseBlock):

    def __init__(self, text_columns,
                 cleansing_hero=None,
                 vectorizer=CountVectorizer(),
                 transformer=TruncatedSVD(n_components=128),
                 transformer2=None,
                 name='html_count_svd',
                 ):
        self.text_columns = text_columns
        self.n_components = transformer.n_components
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.transformer2 = transformer2
        self.name = name
        self.cleansing_hero = cleansing_hero

        self.df = None

    def fit(self, input_df, y=None):
        output_df = pd.DataFrame()
        output_df[self.text_columns] = input_df[self.text_columns].astype(str).fillna('missing')
        features = []
        for c in self.text_columns:
            if self.cleansing_hero is not None:
                output_df[c] = self.cleansing_hero(output_df, c)

            sentence = self.vectorizer.fit_transform(output_df[c])
            feature = self.transformer.fit_transform(sentence)

            if self.transformer2 is not None:
                feature = self.transformer2.fit_transform(feature)

            num_p = feature.shape[1]
            feature = pd.DataFrame(feature, columns=[self.name + str(num_p) + f'_{i:03}' for i in range(num_p)])
            features.append(feature)
        output_df = pd.concat(features, axis=1)
        self.df = output_df

    def transform(self, input_df):
        return self.df


class Doc2VecFeatureTransformer(BaseBlock):

    def __init__(self, text_columns, cleansing_hero=None, params=None, name='doc2vec'):
        self.text_columns = text_columns
        self.cleansing_hero = cleansing_hero
        self.name = name
        self.params = params
        self.df = None

    def fit(self, input_df, y=None):
        dfs = []
        for c in self.text_columns:
            texts = input_df[c].astype(str)
            if self.cleansing_hero is not None:
                texts = self.cleansing_hero(input_df, c)
                texts = [text.split() for text in texts]

            corpus = [TaggedDocument(words=text, tags=[i]) for i, text in enumerate(texts)]
            if self.params is None:
                self.params = {"documents":corpus, "vector_size":128, "min_count":1}
            else:
                self.params["documents"] = corpus
            model = Doc2Vec(**self.params)

            result = np.array([model.infer_vector(text) for text in texts])
            output_df = pd.DataFrame(result)
            output_df.columns = [f'{c}_{self.name}:{i}' for i in range(result.shape[1])]
            dfs.append(output_df)
        output_df = pd.concat(dfs, axis=1)
        self.df = output_df

    def transform(self, dataframe):
        return self.df

# ------------------------------------------------------------------- #
