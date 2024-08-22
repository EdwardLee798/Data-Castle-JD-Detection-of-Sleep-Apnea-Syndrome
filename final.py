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

# 忽略一些不必要的警告信息
warnings.filterwarnings('ignore')

# 设置并获取随机种子，确保结果的可重复性
def seed_obtain(seed=2025):
    np.random.seed(seed)
    random.seed(seed)

seed_obtain()

# 加载数据
def load_data():
    train_X = np.load("/work1/lzy/project/competition/JD/JDCOMP/training_set/train_x.npy")
    train_y = np.load("/work1/lzy/project/competition/JD/JDCOMP/training_set/train_y.npy")
    test_X = np.load("/work1/lzy/project/competition/JD/JDCOMP/test_set_A/test_x_A.npy")
    return train_X, train_y, test_X

# 数据预处理，通过欠采样来处理训练集中的类别不平衡的问题
def preprocess_data(train_X, train_y):
    zero_index = np.where(train_y == 0)[0]
    np.random.shuffle(zero_index)
    # 标签为0的只取4600
    selected_indices = np.concatenate([zero_index[:4600], np.where(train_y != 0)[0]])
    return train_X[selected_indices], train_y[selected_indices]

# 特征工程
def extract_features(data):
    feats = []
    for i in tqdm(range(len(data))):
        data_slice = data[i]
        # 将每个样本的血氧和心率数据取平均值，转化为每秒的平均值
        blood_oxygen = data_slice[0].reshape(-1, 3).mean(axis=1)
        heart_rate = data_slice[1].reshape(-1, 3).mean(axis=1)
        # 将血氧和心率数据构建为一个DataFrame
        origin_feats = pd.DataFrame({'血氧/秒': blood_oxygen, '心率/秒': heart_rate})
        # 增加衍生特征
        add_features(origin_feats)
         # 将衍生特征合并为一个特征集
        feats.append(combine_features(origin_feats))
    return pd.DataFrame(feats)

# 为每个特征列添加衍生特征，包括差分、滚动统计量、自相关等。
def add_features(df):
    for col in df.columns:
        for gap in [1, 2, 4, 8, 16, 30]:
            df[f"{col}_shift{gap}"] = df[col].shift(gap)                  # 移位特征
            df[f"{col}_gap{gap}"] = df[col] - df[f"{col}_shift{gap}"]     # 差分特征
        for window in [3, 5, 10]:
            df[f"{col}_rolling_mean{window}"] = df[col].rolling(window).mean()    # 滚动均值
            df[f"{col}_rolling_std{window}"] = df[col].rolling(window).std()      # 滚动标准差
        for lag in [1, 2, 4, 8, 16, 32]:
            df[f"{col}_autocorr{lag}"] = df[col].autocorr(lag)            # 自相关系数
        freqs, psd = welch(df[col])                                       # 功率谱密度
        df[f"{col}_psd_mean"] = psd.mean()                                # 功率谱密度均值
        df[f"{col}_psd_std"] = psd.std()                                  # 功率谱密度标准差
        df[f"{col}_var"] = df[col].var()                                  # 计算方差
        df[f"{col}_mad"] = np.median(np.abs(df[col] - np.median(df[col])))# 中位数绝对离差/偏差（MAD）

def combine_features(df):
    """
    对每个特征列计算多个统计量，包括均值、最大值、最小值等。
    :param df: 输入的特征 DataFrame
    :return: 统计量列表
    """
    stats = ['mean', 'max', 'min', 'std', 'median', 'skew', 'kurt']
    return [df[col].agg(stats).values for col in df.columns]

 # 使用LightGBM和XGBoost进行模型训练和预测，通过交叉验证评估模型效果。
def train_and_predict(train_feats, test_feats, model_params, num_folds=10, seed=4200):

    # 将训练和测试集的所有数据转换为数值类型，并填充缺失值为0
    train_feats = train_feats.apply(pd.to_numeric, errors='coerce').fillna(0)
    test_feats = test_feats.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 提取特征和标签
    X = train_feats.drop(columns=['label'])
    y = train_feats['label']

    # 初始化模型预测数组
    oof_pred_lgb = np.zeros((len(X), 3), dtype=float)
    oof_pred_xgb = np.zeros((len(X), 3), dtype=float)
    test_pred_pro_lgb = np.zeros((num_folds, len(test_feats), 3), dtype=float)
    test_pred_pro_xgb = np.zeros((num_folds, len(test_feats), 3), dtype=float)

    # 进行分层交叉验证
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    model_lgb = LGBMClassifier(**model_params['lgb'])            # 初始化LightGBM模型
    xgb_params = model_params['xgb']                             # XGBoost模型参数
    
    for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
        print(f"Fold: {fold}")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # 对训练和验证集的数据进行数值化处理，并填充缺失值
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_valid = X_valid.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 训练LightGBM模型
        model_lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      callbacks=[log_evaluation(100), early_stopping(100)])
        # 保存验证集的预测结果
        oof_pred_lgb[valid_index] = model_lgb.predict_proba(X_valid)
        test_pred_pro_lgb[fold] = model_lgb.predict_proba(test_feats)

        # 训练XGBoost模型
        dtrain = DMatrix(X_train, label=y_train)
        dvalid = DMatrix(X_valid, label=y_valid)
        dtest = DMatrix(test_feats)
        evals = [(dvalid, 'eval')]

        xgb_model = xgb_train(xgb_params, dtrain, num_boost_round=10000, evals=evals,
                              early_stopping_rounds=100, verbose_eval=False)
        oof_pred_xgb[valid_index] = xgb_model.predict(dvalid)
        test_pred_pro_xgb[fold] = xgb_model.predict(dtest)
   
    # 综合LightGBM和XGBoost的预测结果
    oof_pred = (oof_pred_lgb + oof_pred_xgb) / 2
    oof_pred_labels = np.argmax(oof_pred, axis=1)
    print(f"Accuracy Score: {accuracy_score(y, oof_pred_labels) * 2}")
    
    # 生成最终测试集的预测结果
    test_pred_pro = (test_pred_pro_lgb + test_pred_pro_xgb).mean(axis=0)
    test_preds = np.argmax(test_pred_pro, axis=1)
    
    return oof_pred, test_preds

# 模型训练过程
train_X, train_y, test_X = load_data()
train_X, train_y = preprocess_data(train_X, train_y)
train_feats = extract_features(train_X)
train_feats['label'] = train_y
test_feats = extract_features(test_X)

# 定义参数
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