from hr_attrition.entity.config_entity import ModelEvaluationConfig
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from hr_attrition.utils.common import save_json
import joblib

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        """
        Evaluate binary classification metrics.
        """
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        return accuracy, precision, recall, f1, roc_auc

    def save_results(self):
        """
        Load test data, make predictions, evaluate metrics, and save results.
        """
        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Separate features and target
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        # Ensure target column is numerical (if categorical)
        if test_y.dtype == 'object':
            test_y = test_y.map({"Yes": 1, "No": 0})  # Adjust mapping based on your data

        # Make predictions
        predicted_labels = model.predict(test_x)

        # Evaluate metrics
        accuracy, precision, recall, f1, roc_auc = self.eval_metrics(test_y, predicted_labels)

        # Save metrics as JSON
        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        save_json(path=self.config.metric_file_name, data=scores)