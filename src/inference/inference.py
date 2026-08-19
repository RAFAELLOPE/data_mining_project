import pandas as pd
import mlflow

class InferenceAgent:
    def __init__(self, config):
        # Load the model from the Model Registry
        model_uri = f"models:/{config.model_name}/Production"
        self.model = mlflow.sklearn.load_model(model_uri)

    def inference(self, 
                  month,
                  productcategory,
                  company_type,
                  company_employees,
                  previous_sales,
                  cpi,
                  unemployment_rate):

        month = int(month)
        if month == 1:
            month = 12
        elif month <= 12:
            month = month -1
        else:
            month = 12
        X = pd.DataFrame(
            {
                'month': str(month),
                'productcategory': productcategory,
                'company_type': company_type,
                'company_employees': company_employees,
                'previous_sales': previous_sales,
                'cpi': cpi,
                'unemployment_rate': unemployment_rate
            }
        )

        prediction = self.model.predict(X)
        return prediction
