import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import lightgbm as lgb
import datetime
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
filepath = 'jinnan_round1_train_20181227.csv'
filepath_test =  'jinnan_round1_testA_20181227.csv'
train_rowdata = pd.read_table(filepath, encoding='gb18030',sep =',')
test_rowdata = pd.read_table(filepath_test, encoding='gb18030',sep =',')

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth',500)
print(train_rowdata.columns)
print(len(train_rowdata))#df.shape[0]
print(train_rowdata.shape[1])

# stats = []
# for col in train_rowdata.columns:
#     stats.append([col, train_rowdata[col].nunique(), train_rowdata[col].isnull().sum()/train_rowdata[col].shape[0], train_rowdata[col].value_counts(normalize=True, dropna=False).values[0], train_rowdata[col].dtype])
#     stats_df = pd.DataFrame(stats,columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
#
# print(stats_df.sort_values('Percentage of missing values', ascending=False).head())

#数据处理
columns =  list(train_rowdata.columns)
# for df in [train_rowdata, test_rowdata]:
#     df.drop(['B3', 'B13', 'A13', 'A18', 'A23'"], axis=1, inplace=True)
print(columns)
remove_ls = ['B3', 'B13', 'A13', 'A18', 'A23']
# for l in ls:
#     columns.remove(l)

# for col in columns:
#     rate = train_rowdata[col].value_counts(normalize=True, dropna=False).values[0]
#     if rate > 0.9:
#         columns.remove(col)

train_rowdata['收率'].loc[train_rowdata['收率']<0.87] = 0.87
# print(train_rowdata[train_rowdata['收率']<0.87]['收率'])
train = train_rowdata[columns].copy()
columns.remove('收率')
test = test_rowdata[columns].copy()

train = train[train["A25"] != "0:00:00"]
train["A25"] = train["A25"].astype("float64")

index = ["样本id"]
time_col = ["A5",'A7', "A9", "A11", "A14", "A16", "A24", "A26",  "B5", "B7"]
period_col = [ "A20","A28","B4","B9", "B10", "B11"]
num_col = []
for col in columns:
    if col not in set(time_col).union(set(period_col).union(index)):
        num_col.append(col)
print(num_col)
num_col1 = ['A1', 'A2', 'A3', 'A4']


# 异常值的处理与预防
print(train.shape[0])
train = train[(train["A6"]>0) & (train["A6"]<60)]
print(train.shape[0])
train = train[(train["A13"]==0.2) ]
print(train.shape[0])
train = train[(train["A17"]>=100) ]
print(train.shape[0])
train = train[(train["A18"]==0.2) ]
print(train.shape[0])
train = train[(train["A21"]>=20) & (train["A21"]<=60)]
print(train.shape[0])
train = train[(train["A22"]>=8) & (train["A22"]<=10)]
print(train.shape[0])
train = train[(train["A23"]==5)]
print(train.shape[0])
train = train[(train["A25"]>=70) & (train["A25"]<=80)]
print(train.shape[0])
train = train[(train["A27"]>=70) & (train["A27"]<=80)]
print(train.shape[0])
train = train[(train["B3"]==3.5)]
print(train.shape[0])
train = train[(train["B8"]<=50)]
print(train.shape[0])
train["B14"][(train["B14"]<=300)]=300
# #处理测试集异常数据
test["A1"][test["A1"]!=300] = 300
test["A4"][test["A4"]!=700] = 700
test["A6"][test["A6"]>40] = 40
test["A19"][test["A19"]>=300] = 300
test["A19"][test["A19"]<=200] = 200
# print("__________")
# print(test["A19"][test["A19"]<200])
test["A25"][test["A25"]>80] = 80
test["A27"][test["A27"]<70] = 70
test["B1"][test["B1"]>400] = 400
test["B14"][test["B14"]>500] = 500
test["B14"][test["B14"]<300] = 300
for col in num_col:
    if col in num_col1:
        train[col].fillna(0, inplace=True)
        test[col].fillna(0, inplace=True)
    else:
        train[col].fillna(stats.mode(train[col].values)[0][0], inplace=True)
        test[col].fillna(stats.mode(test[col].values)[0][0], inplace=True)

target = train['收率']
del train['收率']
data = pd.concat([train, test], ignore_index = True, axis=0)

print(data.shape)

data.fillna(-1, inplace=True)
print("data process finished!")


def tranTime(x):
    try:
        if x==-1:
            return -1
        else:
            l = datetime.datetime.strptime(x, '%H:%M:%S' )
    except:
        print(x)
    else:
        return l

timedata = pd.DataFrame()
for col in time_col:
    try:
        timedata[col] = data[col].apply(tranTime)
    except ValueError:
        print(ValueError)
        print(col)


def split_getTime(item):
    #     try:
    if item == "-1" or str(item) == "":
        return [-1, -1]
    else:
        array = item.split('-')
        #         print(array[0],array[1])
        a1 = datetime.datetime.strptime(array[0], '%H:%M')
        a2 = datetime.datetime.strptime(array[1], '%H:%M')
        #         s[col] = (s[col+"b"] - s[col+"a"]).seconds/3600
        #         return [a0, a1, l.seconds / 3600]
        return [a1, a2]



for col in period_col:
    #     print(train_tempdata[col][train_tempdata[col].isnull()])
    data[col] = data[col].astype(str)
    #     data[col].fillna(0 , inplace = True)
    #     print(data[col][data[col].isnull()], data[col].dtype)
    data[col] = data[col].str.replace(';', ':')
    data[col] = data[col].str.replace('::', ':')
    data[col] = data[col].str.replace('；', ':')
    data[col] = data[col].str.replace('"', ':')
    data[col] = data[col].str.replace(':-', '-')
    data[col] = data[col].str.replace('24:00', '0:00')
    data[col] = data[col].str.replace('19-', '19:00-')
    data[col] = data[col].str.replace('1600', '16:00')
    data[col] = data[col].str.replace('002', '02')
    timedata[col] = data[col].apply(split_getTime)
    timedata[col + "a"] = timedata[col].apply(lambda x: x[0])
    timedata[col + "b"] = timedata[col].apply(lambda x: x[1])
    timedata.drop([col], inplace=True, axis=1)
    #     timedata = pd.concat([timedata, data[col].apply(split_getTime, col=col)], axis=1)#axis
    time_col.extend([col + "a", col + "b"])
# timedata.head()


print(time_col)
def getTime(items):
    if items[0] == -1 or items[1] == -1:
        return -1
    else :
        t = items[1] - items[0]
    return t.seconds/3600
print(np.arange(3))
time_col = ['A5', 'A7', 'A9','A11', 'A14', 'A16', 'A20a', 'A20b', 'A24', 'A26', 'A28a', 'A28b',
            'B4a', 'B4b', 'B5', 'B7', 'B9a', 'B9b', 'B10a', 'B10b', 'B11a', 'B11b']
print(time_col)
duration_col = []
for i in np.arange(len(time_col)-1):
    name = "{}_{}".format(time_col[i], time_col[i+1])
    duration_col.append(name)
    timedata[name] = timedata[[time_col[i],time_col[i+1]]].apply(getTime, axis=1)
# print(timedata.head())
data["AB_time_sum"]  = np.zeros(data.shape[0])
for col in duration_col:
    data["AB_time_sum"] = data["AB_time_sum"] + timedata[col].apply(lambda x: x if x != -1 else 0)

for col in time_col:
    timedata[col] = timedata[col].map(dict(zip(timedata[col].unique(), range(0,timedata[col].nunique()))))
print("time feature finished！")
# data["A6/B14"] = data["A6"] / data['B14']
# data["B6/B14"] = data["B6"] / data['B14']
# data["A10/B14"] = data["A10"] / data['B14']
# data["A17/B14"] = data["A17"] / data['B14']
# data["A6xA5_A7"] = data["A6"] * timedata["A5_A7"]
# data["A8xA7_A9"] = data["A8"] * timedata["A7_A9"]
# data["A10xA9_A11"] = data["A10"] * timedata["A9_A11"]
# data["A12xA13xA11_A14"] = data["A12"]  * timedata["A11_A14"]
# data["A15xA14_A16"] = data["A15"] * timedata["A14_A16"]
# data["A17xA19xA16_A20a"] = data["A17"] * data["A19"]  * timedata["A16_A20a"]
# data["A21xA22xA20a_A20b"] = data["A21"] * data["A22"] * timedata["A20a_A20b"]
# data["A25xA24_A26"] = data["A25"] * timedata["A24_A26"]
# data["A27xA26_A28"] = data["A27"] * timedata["A26_A28a"]
# data["B3xB4a_B4b"] =  data["B3"] * timedata["B4a_B4b"]
# data["B6xB5_B7"] = data["B6"] * timedata["B5_B7"]
data["temperature_sum"] = data["A6"] + data["A8"] + data["A10"] + data["A12"] + data["A15"] + \
                          data["A17"] + data["A21"] + data["A25"] + data["A27"]
# data["temp_A8-A6"] = data["A8"] - data["A6"]
# data["temp_A10-A8"] = data["A10"] - data["A8"]
# data["temp_A12-A10"] = data["A12"] - data["A10"]
# data["temp_A15-A12"] = data["A15"] - data["A12"]
# data["temp_A17-A15"] = data["A17"] - data["A15"]
# data["temp_A21-A17"] = data["A21"] - data["A17"]
# data["temp_A25-A21"] = data["A25"] - data["A21"]
# data["temp_A27-A25"] = data["A27"] - data["A25"]

data["temperature_mean"] = data["temperature_sum"].apply(lambda x: x/9.0)
# data["AB_time_sum"] = data[duration_col].apply(lambda x: x.sum(), axis=1)
data["B1_B2"] = data["B1"] * data["B2"]
data["B12_B13"] = data['B12'] * data['B13']
data["A1+A2+A3"] = data['A1'] + data['A2'] + data['A3']
remove_ls.extend(["B1","B2", 'B12','B13',"A3","A2","A1"])
data["material_sum"] = data["A1+A2+A3"]+data['A4']+data['A19']+data["B1_B2"]+data['B12_B13']
# data["B1_B2_ratio"] = data["B1_B2"]/data["material_sum"]
# data["B12_B13_ratio"] = data["B12_B13"]/data["material_sum"]
# data["A1+A2+A3_ratio"] = data["A1+A2+A3"]/data["material_sum"]
data['material_sum/B14'] = data["material_sum"] / data['B14']
print("factory feature finished!")

# data = pd.concat([data, timedata], ignore_index=True, axis=1)
# columns.remove(features)
features = list(timedata.columns)
columns = list(data.columns)
less_important = ["A9_A11", "A14_A16", "B4b_B5", "A11_A14", "A4", "A5_A7","A24_A26", "A20a_A20b", "A19", "A7_A9"]
for col in features:
    if col in columns:
        columns.remove(col)
for col in period_col:
    if col in columns:
        columns.remove(col)
for col in remove_ls:
    if col in columns:
        columns.remove(col)
for col in less_important:
    if col in columns:
        columns.remove(col)
    if col in duration_col:
        duration_col.remove(col)
# for col in time_col:
#     if col in columns:
#         features.remove(col)
# columns.remove("样本id")
# print(columns)
# for col in columns:
# #     print(col)
#     data[col] = data[col].map(dict(zip(data[col].unique(), range(0,data[col].nunique()))))

for col in features:
    rate = timedata[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.98:
        print(col)
        features.remove(col)

for col in columns:
    rate = data[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.98:
        print(col)
        columns.remove(col)
columns.remove("A8")
id = data["样本id"]
data["样本id"] = data["样本id"].apply(lambda x: int(x.split("_")[1]))
data = pd.concat([data[columns], timedata[duration_col]], axis=1)
train = data[:train.shape[0]]
test = data[train.shape[0]:]
data.head()
# data1.columns
columns = data.columns
print(data.columns)
print(train.shape[0])


def cross_valid(X, y, n_fold, params, ):
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=4058)
    oof_preds = np.zeros(X.shape[0])
    for i, (trn_idx, val_idx) in enumerate(kfold.split(X, y)):
        trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
        test_data = lgb.Dataset(X[val_idx], y[val_idx])
        gbm = lgb.train(params, trn_data ,num_boost_round=10000, valid_sets=[trn_data, test_data], verbose_eval=1000, early_stopping_rounds=100)
        oof_preds[val_idx] = gbm.predict(X[val_idx], num_iteration=gbm.best_iteration)

    loss = mean_squared_error(target,oof_preds)
    return loss/2

def featrueSelect(init_cols):
    params = {
        "learning_rate": 0.01,
        "num_leaves": 31,
        "num_threads": -1,
        "device": "cpu",
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        # "early_stopping_round": 50,
        "lambda_l2": 0.2,
        "metric": "mse",
        "verbosity": -1
    }

    best_cols = init_cols.copy()
    best_score = cross_valid(train[best_cols].values, target.values,5, params)
    print("初始CV score: {:<8.8f}".format(best_score))
    del_col = []
    for f in init_cols:

        best_cols.remove(f)
        score = cross_valid(train[best_cols].values, target.values,5, params)
        diff = best_score - score
        print('-' * 10)
        if diff > 0.0000002:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 有效果,删除！！".format(f, score, best_score))
            del_col.append(f)
            best_score = score
        else:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 没效果,保留！！".format(f, score, best_score))
            best_cols.append(f)
    print('-' * 10)
    print("优化后CV score: {:<8.8f}".format(best_score))
    print(del_col)
    return best_cols

features_all = columns
# columns =  featrueSelect(features_all)
# best_cols = columns
data = data[columns]
train = data[:train.shape[0]]
test = data[train.shape[0]:]
data.head()
# data1.columns
print(data.columns)
print(train.shape[0])

train['target'] = target
train['intTarget'] = pd.cut(train['target'], 5, labels=False)
# print(train['intTarget'])
#B6xB5_B7 A8

train.drop(["intTarget"]+['target'], axis=1, inplace=True)
print(train.shape)
print(test.shape)
train.head()
# train.to_csv("train.csv", index=False )
# test.to_csv("test.csv", index=False)
X_train = train.values
y_train = target.values
X_test = test.values
params = {
    "learning_rate": 0.01,
    "num_leaves": 80,
    "num_threads": -1,
    "device": "cpu",
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    # "early_stopping_round": 50,
    "lambda_l2": 0.2,
    "metric": "mse",
    "verbosity": -1
}
folds = KFold(n_splits=5, shuffle=True, random_state=2019)
predictions_lgb = np.zeros(len(test))
oof_lgb = np.zeros(len(train))

feature_importance_df = pd.DataFrame()
for i, (trn_idx, val_idx) in enumerate(folds. split(X_train, y_train)):
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    test_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
    num_round = 10000
    gbm = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, test_data], verbose_eval=200, early_stopping_rounds=100)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = columns
    fold_importance_df["importance"] = gbm.feature_importance()
    fold_importance_df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    oof_lgb[val_idx] = gbm.predict(X_train[val_idx], num_iteration=gbm.best_iteration)

    predictions_lgb += gbm.predict(test, num_iteration=gbm.best_iteration)
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by = "importance", ascending = False)[:70].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize = (15, 20))
    sns.barplot(x = "importance", y = "feature", data = best_features.sort_values(by = "importance", ascending = False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
#     plt.savefig('datalab/8703/lgbm_importances.png')
display_importances(feature_importance_df)
df_pre_lgb = pd.DataFrame(predictions_lgb / 5, columns=["收率"])
id = pd.DataFrame( id[train.shape[0]:].values, columns=["样本id"])
submit = pd.concat([ id, df_pre_lgb], axis=1)
# submit = test[["样本id", "收率"]]
timeString = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
submit.to_csv("submit1_" + timeString + ".csv", index=False, header=None)

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))