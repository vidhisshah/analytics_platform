from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score

##TODO: Optimize code!!

def rf_algo_r(X_train, X_test, y_train, y_test, html_res):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)#Train the model on training data
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    predictions_rf = [round(value) for value in rf_pred]
    accuracy_rf = accuracy_score(y_test, predictions_rf)
    html_res+="\nAccuracy: " + str(round(accuracy_rf * 100.0,2)) +"%\n"
    html_res+=str(confusion_matrix(y_test, predictions_rf))+"\n"
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    html_res+="Recall: "+str(round(recall_score(y_test, predictions_rf, average=avg),2))+"\n"
    html_res+="Precision: "+str(round(precision_score(y_test, predictions_rf,average=avg)))+"\n"
    print("RF MODEL IN RF", rf)
    return html_res, rf

def rf_algo(X_train, X_test, y_train, y_test, html_res):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)#Train the model on training data
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    # predictions_rf = [round(value) for value in rf_pred]
    predictions_rf = rf_pred
    accuracy_rf = accuracy_score(y_test, predictions_rf)
    html_res+="\nAccuracy: " + str(round(accuracy_rf * 100.0,2)) +"%\n"
    html_res+=str(confusion_matrix(y_test, predictions_rf))+"\n"
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    html_res+="Recall: "+str(round(recall_score(y_test, predictions_rf, average=avg),2))+"\n"
    html_res+="Precision: "+str(round(precision_score(y_test, predictions_rf,average=avg)))+"\n"
    print("RF MODEL IN RF", rf)
    return html_res, rf    


def lgbm_algo_r(X_train, X_test, y_train, y_test, html_res):
    from lightgbm import LGBMRegressor
    lgbm = LGBMRegressor()
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)
    predictions_lgbm = [round(value) for value in y_pred_lgbm]
    accuracy_lgbm = accuracy_score(y_test, predictions_lgbm)
    html_res+="\nAccuracy: " + str(round(accuracy_lgbm * 100.0,2)) +"%\n"
    html_res+=str(confusion_matrix(y_test, predictions_lgbm))+"\n"
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    html_res+="Recall: "+str(round(recall_score(y_test, predictions_lgbm, average=avg),2))+"\n"
    html_res+="Precision: "+str(round(precision_score(y_test, predictions_lgbm,average=avg)))+"\n"
    return html_res, lgbm  

def lgbm_algo(X_train, X_test, y_train, y_test, html_res):
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)
    # predictions_lgbm = [round(value) for value in y_pred_lgbm]
    predictions_lgbm = y_pred_lgbm
    accuracy_lgbm = accuracy_score(y_test, predictions_lgbm)
    html_res+="\nAccuracy: " + str(round(accuracy_lgbm * 100.0,2)) +"%\n"
    html_res+=str(confusion_matrix(y_test, predictions_lgbm))+"\n"
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    html_res+="Recall: "+str(round(recall_score(y_test, predictions_lgbm, average=avg),2))+"\n"
    html_res+="Precision: "+str(round(precision_score(y_test, predictions_lgbm,average=avg)))+"\n"
    return html_res, lgbm  

# XGBoost
def xgb_algo(X_train, X_test, y_train, y_test, html_res):
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    predictions_xgb = [round(value) for value in y_pred_xgb]
    accuracy_xgb = accuracy_score(y_test, predictions_xgb)
    html_res+="\nAccuracy: " + str(round(accuracy_xgb * 100.0,2))+"%\n"
    html_res+=str(confusion_matrix(y_test, predictions_xgb))
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    html_res+="\nRecall: "+ str(round(recall_score(y_test, predictions_xgb, average=avg),2))
    html_res+="\nPrecision: "+ str(round(precision_score(y_test, predictions_xgb, average=avg),2))
    return html_res, xgb


# Gaussian Naive Bayes
def gnb_algo(X_train, X_test, y_train, y_test, html_res):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
    html_res+="\nAccuracy: " + str(round(accuracy_gnb * 100.0, 2))+"%\n"
    html_res+=str(confusion_matrix(y_test,y_pred_gnb))
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    html_res+="\nRecall: "+ str(round(recall_score(y_test, y_pred_gnb, average=avg),2))
    html_res+="\nPrecision: "+ str(round(precision_score(y_test, y_pred_gnb, average=avg),2))
    return html_res, gnb


# Bernoulli Naive Bayes
def bnb_algo(X_train, X_test, y_train, y_test, html_res):
    from sklearn.naive_bayes import BernoulliNB
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    y_pred_bnb = bnb.predict(X_test)
    accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
    html_res+="\nAccuracy: " + str(round(accuracy_bnb * 100.0,2))+"%\n"
    html_res+=str(confusion_matrix(y_test,y_pred_bnb))
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    html_res+="\nRecall: "+ str(round(recall_score(y_test, y_pred_bnb,average=avg),2))
    html_res+="\nPrecision: "+ str(round(precision_score(y_test, y_pred_bnb,average=avg),2))
    return html_res, bnb


# MultinomialNB Naive Bayes
def mnb_algo(X_train, X_test, y_train, y_test, html_res):
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred_mnb = mnb.predict(X_test)
    accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
    html_res+="\nAccuracy: " + str(round(accuracy_mnb * 100.0,2))+"%\n"
    html_res+=str(confusion_matrix(y_test,y_pred_mnb))
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    html_res+="\nRecall: "+ str(round(recall_score(y_test, y_pred_mnb,average=avg),2))
    html_res+="\nPrecision: "+ str(round(precision_score(y_test, y_pred_mnb,average=avg),2))
    return html_res, mnb


# SVM
def svm_algo(X_train, X_test, y_train, y_test, html_res):
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    pred_svc = clf.predict(X_test)
    accuracy_svm = accuracy_score(y_test, pred_svc)
    html_res+="\nAccuracy: " + str(round(accuracy_svm * 100.0,2))+"%\n"
    html_res+=str(confusion_matrix(y_test,pred_svc))
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    html_res+="\nRecall: "+ str(round(recall_score(y_test, pred_svc,average=avg),2))
    html_res+="\nPrecision: "+ str(round(precision_score(y_test, pred_svc,average=avg),2))
    return html_res, clf


# Linear Regression
def linear_reg_algo(X_train, X_test, y_train, y_test, html_res):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X_train, y_train)
    lr_pred = reg.predict(X_test)
    predictions_lr = [round(value) for value in lr_pred]
    accuracy_lr = accuracy_score(y_test, predictions_lr)
    html_res+="\nAccuracy: "+str(round(accuracy_lr,2))+"%\n"
    html_res+=str(confusion_matrix(y_test,predictions_lr))
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    #Error: why does this somtimes think binary class is multiclass?
    # html_res+="\nRecall:"+ str(round(recall_score(y_test,predictions_lr, average=avg),2))
    # html_res+="\nPrecision: "+ str(round(precision_score(y_test, predictions_lr, average=avg),2))
    return html_res, reg


# Logistic Regression
def logistic_reg_algo(X_train, X_test, y_train, y_test, html_res):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    predictions_logr = clf.predict(X_test)
#    predictions_lr = [round(value) for value in lr_pred]
    accuracy_logr = accuracy_score(y_test, predictions_logr)
    html_res+="\nAccuracy: "+str(round(accuracy_logr,2))+"%\n"
    html_res+=str(confusion_matrix(y_test,predictions_logr))
    avg = 'binary' if len(set(y_train.tolist()))==2 else 'micro'
    html_res+="\nRecall:"+ str(round(recall_score(y_test, predictions_logr, average=avg),2))
    html_res+="\nPrecision: "+ str(round(precision_score(y_test, predictions_logr, average=avg),2))
    return html_res, clf
