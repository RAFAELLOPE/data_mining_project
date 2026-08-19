from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD


class Regressor:
    def __init__(self, config):
        self.rf = RandomForestRegressor(random_state=0)
        self.svd = TruncatedSVD(random_state=0)
        self.column_transformer =  ColumnTransformer(
            transformers=[(
                'onehot',  
                OneHotEncoder(handle_unknown='ignore'), 
                config.categorical_features
            ),
            (
                'scale',
                StandardScaler(),
                config.numeric_features
            )],
            remainder='passthrough'
        )

        self.model = Pipeline([
            ('column_transformer', self.column_transformer),
            ('svd', self.svd),
            ('regressor', self.rf)
        ])

    def get_model(self):
        return self.model
