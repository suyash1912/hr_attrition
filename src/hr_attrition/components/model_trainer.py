




from hr_attrition.entity.config_entity import ModelTrainingConfig
import pandas as pd
import os
from hr_attrition.utils.logging import logger
from sklearn.linear_model import LogisticRegression
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def train(self):
        # Load the data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Separate features and target
        train_x = train_data.drop("Attrition", axis=1)
        test_x = test_data.drop("Attrition", axis=1)
        train_y = train_data["Attrition"]
        test_y = test_data["Attrition"]

        # Initialize and train logistic regression model
        model = LogisticRegression(
            C=self.config.C,
            penalty=self.config.penalty,
            solver=self.config.solver,
            class_weight=self.config.class_weight,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state
        )
        model.fit(train_x, train_y)

        # Save model to disk
        joblib.dump(model, self.config.model_path)

        logger.info(f"Model trained and saved at {self.config.model_path}")
