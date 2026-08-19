from src.loader.loader import DataLoader
from src.trainer.trainer import Regressor

from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow

import warnings
warnings.filterwarnings('ignore')


class Experiment:
    def __init__(self, config):
        
        self.experiment_name = config.experiment_name
        self.run_name = config.run_name

        # Define the hyperparameter search space for GridSearchCV
        self.svd_components = config.svd_component_search_space
        self.rf_estimators = config.rf_estimators_search_space
        self.param_grid = {
            "svd__n_components": self.svd_components,
            "regressor__n_estimators": self.rf_estimators,
        }

        # Number of cross-validation folds
        self.cv_folds = config.CV_folds

        self.loader = DataLoader(config)
        self.X_train, self.X_test, self.y_train, self.y_test = self.loader.get_train_test_data()
        self.model = Regressor().get_model()
        self.best_model = None
        self.best_params = None





