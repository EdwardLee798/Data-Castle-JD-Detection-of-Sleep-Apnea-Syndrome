import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from scipy.signal import welch
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from xgboost import DMatrix, train as xgb_train
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import random

# Ignore some unnecessary warning messages
warnings.filterwarnings('ignore')

# Set and obtain random seed to ensure reproducibility of results
def seed_obtain(seed=2025):
    np.random.seed(seed)
    random.seed(seed)

seed_obtain()

# Load data
def load_data():
    train_X = np.load("/work1/lzy/project/competition/JD/JDCOMP/training_set/train_x.npy")
    train_y = np.load("/work1/lzy/project/competition/JD/JDCOMP/training_set/train_y.npy")
    test_X = np.load("/work1/lzy/project/competition/JD/JDCOMP/test_set_A/test_x_A.npy")
    return train_X, train_y, test_X

# Data preprocessing, handling class imbalance in the training set by undersampling
def preprocess_data(train_X, train_y):
    zero_index = np.where(train_y == 0)[0]
    np.random.shuffle(zero_index)
    # Only take 4600 samples with label 0
    selected_indices = np.concatenate([zero_index[:4600], np.where(train_y != 0)[0]])
    return train_X[selected_indices], train_y[selected_indices]

# Feature engineering
def extract_features(data):
    feats = []
    for i in tqdm(range(len(data))):
        data_slice = data[i]
        # Take the average value of blood oxygen and heart rate data for each sample, converted to the average per second
        blood_oxygen = data_slice[0].reshape(-1, 3).mean(axis=1)
        heart_rate = data_slice[1].reshape(-1, 3).mean(axis=1)
        # Construct a DataFrame with blood oxygen and heart rate data
        origin_feats = pd.DataFrame({'Blood Oxygen/sec': blood_oxygen, 'Heart Rate/sec': heart_rate})
        # Add derived features
        add_features(origin_feats)
        # Combine the derived features into a feature set
        feats.append(combine_features(origin_feats))
    return pd.DataFrame(feats)

# Add derived features for each feature column, including differences, rolling statistics, autocorrelations, etc.
def add_features(df):
    for col in df.columns:
        for gap in [1, 2, 4, 8, 16, 30]:
            df[f"{col}_shift{gap}"] = df[col].shift(gap)                  # Shifted features
            df[f"{col}_gap{gap}"] = df[col] - df[f"{col}_shift{gap}"]     # Difference features
        for window in [3, 5, 10]:
            df[f"{col}_rolling_mean{window}"] = df[col].rolling(window).mean()    # Rolling mean
            df[f"{col}_rolling_std{window}"] = df[col].rolling(window).std()      # Rolling standard deviation
        for lag in [1, 2, 4, 8, 16, 32]:
            df[f"{col}_autocorr{lag}"] = df[col].autocorr(lag)            # Autocorrelation coefficient
        freqs, psd = welch(df[col])                                       # Power spectral density
        df[f"{col}_psd_mean"] = psd.mean()                                # Power spectral density mean
        df[f"{col}_psd_std"] = psd.std()                                  # Power spectral density standard deviation
        df[f"{col}_var"] = df[col].var()                                  # Calculate variance
        df[f"{col}_mad"] = np.median(np.abs(df[col] - np.median(df[col])))# Median Absolute Deviation (MAD)

def combine_features(df):
    """
    Calculate multiple statistics for each feature column, including mean, max, min, etc.
    :param df: Input feature DataFrame
    :return: List of statistics
    """
    stats = ['mean', 'max', 'min', 'std', 'median', 'skew', 'kurt']
    return [df[col].agg(stats).values for col in df.columns]

# Train and predict using LightGBM and XGBoost, evaluating model performance through cross-validation.
def train_and_predict(train_feats, test_feats, model_params, num_folds=10, seed=4200):

    # Convert all data in the training and test sets to numeric types and fill missing values with 0
    train_feats = train_feats.apply(pd.to_numeric, errors='coerce').fillna(0)
    test_feats = test_feats.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Extract features and labels
    X = train_feats.drop(columns=['label'])
    y = train_feats['label']

    # Initialize model prediction arrays
    oof_pred_lgb = np.zeros((len(X), 3), dtype=float)
    oof_pred_xgb = np.zeros((len(X), 3), dtype=float)
    test_pred_pro_lgb = np.zeros((num_folds, len(test_feats), 3), dtype=float)
    test_pred_pro_xgb = np.zeros((num_folds, len(test_feats), 3), dtype=float)

    # Perform stratified cross-validation
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    model_lgb = LGBMClassifier(**model_params['lgb'])            # Initialize LightGBM model
    xgb_params = model_params['xgb']                             # XGBoost model parameters
    
    for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
        print(f"Fold: {fold}")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # Convert data for the training and validation sets to numeric types and fill missing values
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_valid = X_valid.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Train the LightGBM model
        model_lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      callbacks=[log_evaluation(100), early_stopping(100)])
        # Save the prediction results for the validation set
        oof_pred_lgb[valid_index] = model_lgb.predict_proba(X_valid)
        test_pred_pro_lgb[fold] = model_lgb.predict_proba(test_feats)

        # Train the XGBoost model
        dtrain = DMatrix(X_train, label=y_train)
        dvalid = DMatrix(X_valid, label=y_valid)
        dtest = DMatrix(test_feats)
        evals = [(dvalid, 'eval')]

        xgb_model = xgb_train(xgb_params, dtrain, num_boost_round=10000, evals=evals,
                              early_stopping_rounds=100, verbose_eval=False)
        oof_pred_xgb[valid_index] = xgb_model.predict(dvalid)
        test_pred_pro_xgb[fold] = xgb_model.predict(dtest)
   
    # Combine the prediction results from LightGBM and XGBoost
    oof_pred = (oof_pred_lgb + oof_pred_xgb) / 2
    oof_pred_labels = np.argmax(oof_pred, axis=1)
    print(f"Accuracy Score: {accuracy_score(y, oof_pred_labels) * 2}")
    
    # Generate the final prediction results for the test set
    test_pred_pro = (test_pred_pro_lgb + test_pred_pro_xgb).mean(axis=0)
    test_preds = np.argmax(test_pred_pro, axis=1)
    
    return oof_pred, test_preds

# Model training process
train_X, train_y, test_X = load_data()
train_X, train_y = preprocess_data(train_X, train_y)
train_feats = extract_features(train_X)
train_feats['label'] = train_y
test_feats = extract_features(test_X)

# Define parameters
model_params = {
    'lgb': {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "max_depth": 10, 
        "learning_rate": 0.01, 
        "n_estimators": 15000,  
        "colsample_bytree": 0.7,  
        "subsample": 0.8, 
        "verbose": -1,
        "random_state": 2024,
        "reg_alpha": 0.1,  
        "reg_lambda": 2,  
        "extra_trees": True,
        'num_leaves': 100,  
        "max_bin": 255,
    },
    'xgb': {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 10,  
        "learning_rate": 0.01,  
        "colsample_bytree": 0.7,
        "subsample": 0.8,  
        "random_state": 2024,
        "reg_alpha": 0.1,  
        "reg_lambda": 2,  
        "min_child_weight": 3  
    }
}
oof_pred, test_preds = train_and_predict(train_feats, test_feats, model_params)
submission = pd.read_csv("/work1/lzy/project/competition/JD/JDCOMP/test_set_A/submit_example_A.csv")
submission['label'] = test_preds
submission.to_csv("baseline.csv", index=None)
