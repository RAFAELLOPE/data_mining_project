class Config:
  """
  Holds configuration parameters
  """
  def __init__(self):
    self.experiment_name = "Random_Forest_Regressor"
    self.run_name = "19082024_rf"
    self.file_path = 'C:\\Users\\34616\\Documents\\Cursos\\Data_Mining\\data_mining_project\\data\\gold\\purchases.csv'
    self.mlflow_dir = 'sqlite:///c:/Users/34616/Documents/Cursos/Data_Mining/data_mining_project/data/mlflow.db'

    self.categorical_features = ['productcategory', 'company_type', 'company_employees', 'month']
    self.numeric_features = ['previous_sales', 'cpi', 'unemployment_rate']
    
    self.test_size = 0.2
    self.svd_component_search_space = [5, 10, 15, 20]
    self.rf_estimators_search_space = [10, 50, 100]
    
    self.CV_folds = 5
    self.test_results_dir = '.C:\\Users\\34616\\Documents\\Cursos\\Data_Mining\\data_mining_project\\data\\gold\\'
    