
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, config):
        self.test_size = config.test_size

        data = pd.read_csv(
            config.file_path
        )

        self.df = data.drop(
            'date',
            axis=1
        )

    def get_train_test_data(self):
        X = self.df.drop('sales', axis=1)
        y = self.df.sales

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    