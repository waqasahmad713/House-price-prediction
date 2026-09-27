# House Price Prediction

Predicts California housing prices from `housing.csv`. The script builds room and household features, keeps the income mix in the train and test split, then compares gradient boosting and random forest. The best model is saved as `best_housing_model.joblib`.

## Run

```bash
pip install numpy pandas scikit-learn joblib
python3 house_prediction.py
```
