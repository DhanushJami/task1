import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import os

# Load data from CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess data using pipelines
def preprocess_data(df):
    X = df.copy()

    # Drop identifier or high-cardinality columns
    identifier_cols = [col for col in X.columns if 'id' in col.lower() and X[col].nunique() > 100]
    high_card_cat_cols = [col for col in X.select_dtypes(include=['object', 'category']).columns if X[col].nunique() > 50]
    X.drop(columns=identifier_cols + high_card_cat_cols, inplace=True, errors='ignore')

    # Separate types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    # Fit and transform
    processed_array = preprocessor.fit_transform(X)
    processed_columns = numeric_cols + list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols))

    return pd.DataFrame(processed_array, columns=processed_columns)

# Main ETL pipeline
def etl_pipeline(file_path):
    # Load
    df = load_data(file_path)

    # Show preview
    print("Initial Data Sample:\n")
    print(df.head())

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Preprocess
    processed_train = preprocess_data(train_df)
    processed_test = preprocess_data(test_df)

    # Save
    processed_train.to_csv('processed_train_data.csv', index=False)
    processed_test.to_csv('processed_test_data.csv', index=False)

    print("\nETL Pipeline Completed. Files saved as 'processed_train_data.csv' and 'processed_test_data.csv'")

# Entry point
if __name__ == "__main__":
    file_path = "Balaji Fast Food Sales.csv"
    if os.path.exists(file_path):
        etl_pipeline(file_path)
    else:
        print(f"❌ File not found: {file_path}")
