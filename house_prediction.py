import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the California housing dataset from a local CSV path."""
    return pd.read_csv(csv_path)


def data_understanding(df: pd.DataFrame) -> None:
    """Print summary statistics, information, and missing value counts."""
    print('STEP 2: DATA UNDERSTANDING')
    print('-' * 60)
    print('Shape:', df.shape)
    print('\nData types and non-null counts:')
    print(df.info())
    print('\nSummary statistics:')
    print(df.describe().T)
    print('\nMissing values:')
    print(df.isna().sum())
    print('\n')


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from existing numeric columns."""
    output = df.copy()
    output['rooms_per_household'] = output['total_rooms'] / output['households']
    output['bedrooms_per_room'] = output['total_bedrooms'] / output['total_rooms']
    output['population_per_household'] = output['population'] / output['households']
    return output


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data with stratification on income category to preserve distribution."""
    df = df.copy()
    # Drop rows without target values because we cannot train on missing labels.
    df = df.dropna(subset=['median_house_value'])
    # Impute missing income values before creating stratification categories.
    df['median_income'] = df['median_income'].fillna(df['median_income'].median())

    df['income_cat'] = pd.cut(
        df['median_income'],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(df, df['income_cat']):
        train_set = df.iloc[train_index].drop(columns=['income_cat'])
        test_set = df.iloc[test_index].drop(columns=['income_cat'])
    return train_set, test_set


def handle_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Impute missing values using median values computed from training data only."""
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_imputed, X_test_imputed


def evaluate_model(model, X, y, dataset_name: str):
    """Return evaluation metrics for a regression model."""
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    print(f'{dataset_name} results for {model.__class__.__name__}:')
    print(f'  R2 Score: {r2:.4f}')
    print(f'  MAE: {mae:.2f}')
    print(f'  RMSE: {rmse:.2f}')
    print('-' * 50)
    return {'r2': r2, 'mae': mae, 'rmse': rmse}


def tune_model(model, param_distributions, X, y, cv: int = 3, n_iter: int = 12):
    """Run randomized hyperparameter search and return the fitted search object."""
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X, y)
    return search


def main() -> None:
    csv_path = Path(__file__).parent / 'housing.csv'
    print('STEP 1: LOAD DATASET')
    print('-' * 60)
    data = load_data(csv_path)
    print('Loaded dataset from:', csv_path)
    print('\n')

    data_understanding(data)

    train_set, test_set = split_data(data)
    print('STEP 3: TRAIN-TEST SPLIT')
    print('-' * 60)
    print('Train set shape:', train_set.shape)
    print('Test set shape :', test_set.shape)
    print('\n')

    train_labels = train_set['median_house_value'].copy()
    test_labels = test_set['median_house_value'].copy()
    X_train = train_set.drop(columns=['median_house_value'])
    X_test = test_set.drop(columns=['median_house_value'])

    X_train_imputed, X_test_imputed = handle_missing_values(X_train, X_test)

    X_train_fe = add_features(X_train_imputed)
    X_test_fe = add_features(X_test_imputed)
    print('STEP 4: FEATURE ENGINEERING')
    print('-' * 60)
    print('New features added: rooms_per_household, bedrooms_per_room, population_per_household')
    print('Training data shape after feature engineering:', X_train_fe.shape)
    print('Test data shape after feature engineering    :', X_test_fe.shape)
    print('\n')

    print('STEP 5: FEATURE SCALING')
    print('-' * 60)
    print('Tree-based models like Random Forest and Gradient Boosting do not require feature scaling.')
    print('We keep raw numeric values and engineered ratios for interpretability and robustness.')
    print('\n')

    print('STEP 6: MODEL TRAINING')
    print('-' * 60)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)

    rf_model.fit(X_train_fe, train_labels)
    gb_model.fit(X_train_fe, train_labels)
    print('Trained Random Forest Regressor and Gradient Boosting Regressor.')
    print('\n')

    print('STEP 7: EVALUATION ON TRAIN SET')
    print('-' * 60)
    evaluate_model(rf_model, X_train_fe, train_labels, 'Train set')
    evaluate_model(gb_model, X_train_fe, train_labels, 'Train set')

    print('STEP 8: EVALUATION ON TEST SET')
    print('-' * 60)
    results_rf = evaluate_model(rf_model, X_test_fe, test_labels, 'Test set')
    results_gb = evaluate_model(gb_model, X_test_fe, test_labels, 'Test set')

    print('STEP 9: MODEL COMPARISON')
    print('-' * 60)
    if results_gb['r2'] > results_rf['r2']:
        best_model_name = 'Gradient Boosting Regressor'
        best_model = gb_model
        best_results = results_gb
    else:
        best_model_name = 'Random Forest Regressor'
        best_model = rf_model
        best_results = results_rf

    print(f'Best model: {best_model_name}')
    print(f'Best test R2: {best_results["r2"]:.4f}')
    print(f'Best test MAE: {best_results["mae"]:.2f}')
    print(f'Best test RMSE: {best_results["rmse"]:.2f}')
    print('\n')

    print('STEP 10: MODEL TUNING AND IMPROVEMENTS')
    print('-' * 60)
    print('Tuning the best candidate model with randomized search.')
    gb_param_distributions = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 4, 6]
    }
    gb_search = tune_model(gb_model, gb_param_distributions, X_train_fe, train_labels, cv=3, n_iter=15)
    print('Best Gradient Boosting parameters:', gb_search.best_params_)

    tuned_gb = gb_search.best_estimator_
    print('\nEvaluating tuned Gradient Boosting on the test set:')
    tuned_results_gb = evaluate_model(tuned_gb, X_test_fe, test_labels, 'Tuned Gradient Boosting test set')

    if tuned_results_gb['r2'] > best_results['r2']:
        best_model = tuned_gb
        best_model_name = 'Tuned Gradient Boosting Regressor'
        best_results = tuned_results_gb
        print('Tuned model improved the test R2 and is now the selected best model.')
    else:
        print('Tuned model did not beat the prior best model on the test set.')

    print('\nFinal best model:', best_model_name)
    print(f'Final best test R2: {best_results["r2"]:.4f}')
    print(f'Final best test MAE: {best_results["mae"]:.2f}')
    print(f'Final best test RMSE: {best_results["rmse"]:.2f}')
    print('\n')

    model_path = Path(__file__).parent / 'best_housing_model.joblib'
    joblib.dump({'model': best_model, 'feature_columns': X_train_fe.columns.tolist()}, model_path)
    print('Saved best model to:', model_path)


if __name__ == '__main__':
    main()
