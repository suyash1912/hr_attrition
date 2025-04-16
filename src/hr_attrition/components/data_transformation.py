
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from hr_attrition import logger
from hr_attrition.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_data(self):
        data = pd.read_csv(self.config.data_path)
        logger.info("Data loaded successfully.")
        return data

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Map Attrition to binary: Yes -> 1, No -> 0
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
        logger.info("Mapped target 'Attrition' to binary.")

        # Separate features and target
        target = df["Attrition"]
        X = df.drop(columns=["Attrition"])

        # Separate numeric and categorical columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        logger.info(f"Numerical columns: {num_cols}")
        logger.info(f"Categorical columns: {cat_cols}")

        # Pipelines
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("power_transform", PowerTransformer(method="yeo-johnson"))
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ])

        logger.info("Fitting and transforming the features.")
        X_processed = preprocessor.fit_transform(X)

        # Get column names
        cat_encoded_cols = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols)
        feature_names = num_cols + list(cat_encoded_cols)

        X_processed = pd.DataFrame(X_processed, columns=feature_names)

        # Add target back
        df_final = pd.concat([X_processed, target.reset_index(drop=True)], axis=1)

        return df_final

    def train_test_spliting(self):
        df = self.load_data()
        df_processed = self.preprocess_data(df)

        train, test = train_test_split(
            df_processed,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df_processed["Attrition"]
        )

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Data preprocessed and split with stratification on binary target.")
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

        print("Train shape:", train.shape)
        print("Test shape:", test.shape)
