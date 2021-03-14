import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import ListedColormap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.preprocessing import MinMaxScaler

from load_data import load_odds_and_538_data
from utils.slize_games_into_chunks import chunk_games

pd.set_option('display.width', None)

df_538_eval = load_odds_and_538_data(leagues=['D1', 'E0'])
df_538_eval['spi_diff'] = df_538_eval['spi1'] - df_538_eval['spi2']

gd_features = ['odds_diff', 'spi_diff']
gd_label = 'goal_diff'
# throw out draws because it is not worth picking them
train_split = train_test_split(df_538_eval.index, test_size=0.30, random_state=1)
df_train = df_538_eval.copy(deep=True)

relevant_labels = (df_train[gd_label].value_counts() > (0.01 * df_train.shape[0]))
relevant_labels = relevant_labels.loc[relevant_labels == True].index
df_train = df_train.loc[[x in relevant_labels for x in df_train[gd_label]], :]

df_test = df_538_eval.loc[[x in train_split[1] for x in df_538_eval.index], :].copy(deep=True)
df_test = df_test.loc[[x in relevant_labels for x in df_test[gd_label]], :]

y_gd_train = df_train[gd_label].values.astype(int)
y_gd_test = df_test[gd_label].values.astype(int)

gd_scaler = PCA(n_components=len(gd_features))

gd_scaler.fit(df_train[gd_features].values)
gd_train_scaled = gd_scaler.transform(df_train[gd_features].values)
gd_test_scaled = gd_scaler.transform(df_test[gd_features].values)

minmax_scaler = MinMaxScaler()
minmax_scaler.fit(gd_train_scaled)
gd_train_scaled = minmax_scaler.transform(gd_train_scaled)
gd_test_scaled = minmax_scaler.transform(gd_test_scaled)

classes = np.unique(y_gd_train)
dict_goal_diff_to_class = {classes[idx]: idx for idx in range(len(classes))}
dict_class_to_goal_diff = {idx: classes[idx] for idx in range(len(classes))}

dtrain = xgb.DMatrix(gd_train_scaled, label=[dict_goal_diff_to_class[x] for x in y_gd_train])
dtest = xgb.DMatrix(gd_test_scaled, label=[dict_goal_diff_to_class[x] for x in y_gd_test])

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 50
param = {'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': len(classes), 'max_depth': 3}
bst = xgb.train(param, dtrain, num_round, evallist)
ypred_xgb = bst.predict(dtrain)
df_pred_prob = pd.DataFrame(ypred_xgb, columns=classes)
df_pred_result = pd.DataFrame(df_pred_prob.idxmax(axis=1), columns=[gd_label])

calibrated_clf = CalibratedClassifierCV(base_estimator=ComplementNB(), cv=5)
calibrated_clf.fit(gd_train_scaled, [dict_goal_diff_to_class[x] for x in y_gd_train])
yprob_comp = calibrated_clf.predict_proba(gd_train_scaled)
df_pred_comp_prob = pd.DataFrame(yprob_comp, columns=classes)
ypred_comp = calibrated_clf.predict(gd_train_scaled)
df_pred_comp_result = pd.DataFrame(df_pred_comp_prob.idxmax(axis=1), columns=[gd_label])

calibrated_clf = CalibratedClassifierCV(base_estimator=GaussianNB(), cv=5)
calibrated_clf.fit(gd_train_scaled, [dict_goal_diff_to_class[x] for x in y_gd_train])
yprob_gaussian = calibrated_clf.predict_proba(gd_train_scaled)
df_pred_gaussian_prob = pd.DataFrame(yprob_gaussian, columns=classes)
ypred_gaussian = calibrated_clf.predict(gd_train_scaled)
df_pred_gaussian_result = pd.DataFrame(df_pred_gaussian_prob.idxmax(axis=1), columns=[gd_label])

plt.figure(figsize=(12, 6))
for idx_res in range(9):
    plt.subplot(331 + idx_res)
    actual_scores = np.array(y_gd_train) == classes[idx_res]

    pred_gnb_scores = [dict_class_to_goal_diff[x] for x in ypred_gaussian] == classes[idx_res]
    f1_gnb = np.round(f1_score(actual_scores, pred_gnb_scores), 3)
    df_result = pd.DataFrame(yprob_gaussian[:, idx_res], columns=['pred_res']).join(
        pd.DataFrame(actual_scores, columns=['true_res']))
    df_chunked = chunk_games(df_result, chunk_by='pred_res', chunk_size=50)
    df_mean = df_chunked.groupby(by='chunk').mean()
    plt.plot(df_mean['pred_res'], df_mean['true_res'], label='gnb (f1: ' + str(f1_gnb) + ')')


    pred_comb_scores = [dict_class_to_goal_diff[x] for x in ypred_comp] == classes[idx_res]
    f1_comb = np.round(f1_score(actual_scores, pred_comb_scores), 3)
    df_result = pd.DataFrame(yprob_comp[:, idx_res], columns=['pred_res']).join(
        pd.DataFrame(actual_scores, columns=['true_res']))
    df_chunked = chunk_games(df_result, chunk_by='pred_res', chunk_size=50)
    df_mean = df_chunked.groupby(by='chunk').mean()
    plt.plot(df_mean['pred_res'], df_mean['true_res'], label='comb (f1: ' + str(f1_comb) + ')')


    df_result = pd.DataFrame(ypred_xgb[:, idx_res], columns=['pred_res']).join(
        pd.DataFrame(actual_scores, columns=['true_res']))
    df_chunked = chunk_games(df_result, chunk_by='pred_res', chunk_size=50)
    df_mean = df_chunked.groupby(by='chunk').mean()
    plt.plot(df_mean['pred_res'], df_mean['true_res'], label='xgb')

    plt.plot([0, 0.4], [0, 0.4], label='perfect')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(str(classes[idx_res]) + ' | sample size: ' + str(sum(actual_scores)) + ' of ' + str(len(actual_scores)))
    plt.legend()
plt.show()

# preprocess dataset, split into training and test part
X_test = gd_test_scaled
X_train = gd_train_scaled

x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
h = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
d_xy = xgb.DMatrix(np.c_[xx.ravel(), yy.ravel()])
Z = bst.predict(d_xy)
df_zraw = pd.DataFrame(Z, columns=classes)
common_predictions = classes[:9]
for idx_pred in range(len(common_predictions)):
    curr_result = common_predictions[idx_pred]
    plt.subplot(331 + idx_pred)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    Z = df_zraw[curr_result].values.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_gd_train == curr_result, cmap=cm_bright, edgecolors='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_gd_test == curr_result, cmap=cm_bright, alpha=0.6, edgecolors='k')
    plt.title('XGB: ' + str(curr_result))
plt.show()

d_xy = xgb.DMatrix(np.c_[xx.ravel(), yy.ravel()])
Z = calibrated_clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
df_zraw = pd.DataFrame(Z, columns=classes)
common_predictions = classes[:9]
for idx_pred in range(len(common_predictions)):
    curr_result = common_predictions[idx_pred]
    plt.subplot(331 + idx_pred)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    Z = df_zraw[curr_result].values.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_gd_train == curr_result, cmap=cm_bright, edgecolors='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_gd_test == curr_result, cmap=cm_bright, alpha=0.6, edgecolors='k')
    plt.title('GaussianNB: ' + str(curr_result))
plt.show()
