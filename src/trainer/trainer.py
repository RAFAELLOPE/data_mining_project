from src.loader.loader import DataLoader
from src.trainer.trainer import Regressor

import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow

import warnings
warnings.filterwarnings('ignore')


class Experiment:
    def __init__(self, config):
        
        self.experiment_name = config.experiment_name
        self.run_name = config.run_name
        self.mlflow_dir = config.mlflow_dir

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

        # Create a timestamped output folder for experiment artifacts
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        self.best_model = None
        self.best_params = None

    def train(self):
        mlflow.set_tracking_uri(self.mlflow_dir)
        mlflow.autolog()
        mlflow.set_experiment(self.experiment_name)
        current_experiment=dict(mlflow.get_experiment_by_name(self.experiment_name))
        experiment_id=current_experiment['experiment_id']
        with mlflow.start_run(experiment_id=experiment_id, run_name="cv_random_forest"):
            cv = GridSearchCV(
                self.model, 
                self.param_grid, 
                cv=self.cv_folds, 
                scoring='r2')

            # Compute model with best recall
            cv.fit(self.X_train, self.y_train)
            self.best_model = cv.best_estimator_
            self.best_params = cv.best_params_

            mlflow.sklearn.log_model(
                self.best_model,
                artifact_path="model",
                registered_model_name="RandomForestRegressor")

    def evaluate(self):
        current_experiment=dict(mlflow.get_experiment_by_name(self.experiment_name))
        experiment_id=current_experiment['experiment_id']
        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            order_by=["start_time DESC"],
        )

        run_id = runs.iloc[0]["run_id"]

        with mlflow.start_run(run_id=run_id):
            y_pred = self.best_model.predict(self.X_test)
            r2 = r2_score(y_true=self.y_test, y_pred=y_pred)
            mse =  mean_squared_error(y_true=self.y_test, y_pred=y_pred)
            mae = mean_absolute_error(y_true=self.y_test, y_pred=y_pred)
            self.plot_feature_importance()

            # Log metrics
            mlflow.log_metric("test_r2", r2)
            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_mae", mae)

            # Log Feature importance
            mlflow.log_artifact(
                os.path.join(self.out_dir, "feature_importance.png"),
                artifact_path="feature_importance"
            )

    def plot_feature_importance(self, top_n=5):
        # Feature importance
        feature_importance = pd.DataFrame({
            "feature": self.model.feature_names_in_,
            "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        plot_data = feature_importance.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(
            plot_data["feature"][::-1],
            plot_data["importance"][::-1]
        )
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.out_dir,
                "feature_importance.png"
            ), 
            dpi=150
        )
        plt.close()

    def run(self):
        self.train()
        self.evaluate()









