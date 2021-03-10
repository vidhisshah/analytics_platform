#!/usr/bin/env python
#coding: utf-8

import pandas  as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_boston, load_iris, load_wine
from flask import Flask, render_template, url_for, request
from ml_algos import rf_algo, lgbm_algo, xgb_algo, gnb_algo, bnb_algo, mnb_algo, svm_algo, linear_reg_algo, logistic_reg_algo


app = Flask(__name__)

#Read CSV
def read_file(filepath):
    return pd.read_csv(filepath)


# Creating functions for every algo, separate modules to allow paramater tuning for each in future

# Execute all algos
#TODO: Add intelligence as to categorical or numeric prediction - OR Classification or regression problem
def execut(X_train, X_test, y_train, y_test):
    # global html_res 
    print("IN EXECUTION")
    html_res = "\n\n TN   FP\n FN    TP"
    html_res +="\n\n***Random Forest***\n"
    html_res, rf_model = rf_algo(X_train, X_test, y_train, y_test, html_res)   
    html_res+="\n\n***Light GBM***\n"
    html_res, lgbm_model = lgbm_algo(X_train, X_test, y_train, y_test, html_res)
    html_res+="\n\n***XG Boost***\n"
    html_res, xgb_model = xgb_algo(X_train, X_test, y_train, y_test, html_res)
    html_res+="\n\n***SVM***\n"
    html_res, svm_model = svm_algo(X_train, X_test, y_train, y_test, html_res)
    html_res+="\n\n***Gaussian Naive Bayes***\n"
    html_res, gnb_model = gnb_algo(X_train, X_test, y_train, y_test, html_res)
    html_res+="\n\n***Multinomial Naive Bayes***\n"
    html_res, mnb_model = mnb_algo(X_train, X_test, y_train, y_test, html_res)
    # html_res+="\n\n***Linear Regression***\n"
    # html_res, lr_model = linear_reg_algo(X_train, X_test, y_train, y_test, html_res)
    # html_res+="\n\n***Logistic Regression***\n"
    # html_res, logr_model = logistic_reg_algo(X_train, X_test, y_train, y_test, html_res)
    ## SAVE MODELS TO USE FOR PREDICTION
    # pickle.dump(rf_model,open('rf_model.pkl','wb'))
    # pickle.dump(gnb_model, open('gnb_model.pkl','wb'))
    # pickle.dump(mnb_model, open('mnb_model.pkl', 'wb'))
    #FOR REGRESSION
    # return html_res, rf_model_r, lgbm_model, xgb_model, svm_model, gnb_model, mnb_model, lr_model, logr_model
    #FOR CLASSIFICATION
    return html_res, rf_model, lgbm_model, xgb_model, svm_model, gnb_model, mnb_model, "", ""

def concat_files(files_list):
    #can later convert to checking number of files of csv or xlsx in folder
    print("Files List:", files_list)
    dfs_list=[]
    # n = int(input("Enter number of files: "))
    if len(files_list) > 1:        
        for i in files_list:
            dfs_list.append(read_file(i))
        df = pd.concat(dfs_list, ignore_index=True)
    elif len(files_list) == 1:
        print("Reading file")
        df = pd.read_csv(files_list[0])
    else:
        print("Error")
        df=pd.DataFrame()    
    return df

def def_X_y(cols_to_drop, target):
    global df
    # cols_to_drop = []
    # n = input("Enter number of columns to drop: ")
    # for i in range(int(n)):
    #     cols_to_drop.append(input("Enter column name: "))
    # X = df.drop(cols_to_drop,axis=1)
    # X.fillna(99, inplace=True)
    # to_be_predicted=""
    # to_be_predicted = input('Select value to be predicted: ')
    # print(to_be_predicted)
    # if to_be_predicted not in df.columns:
    #     Y = df[df.columns[-1]]
    # else:
    #     Y = df[to_be_predicted]
    X=df.drop(cols_to_drop, axis=1)
    X.fillna(9, inplace=True)
    Y=df[target]
    return X,Y    

def split(X,y, split_ratio=0.3):
    # try:
    #     split_ratio = float(input('Enter split ratio\neg - 0.3 implies 70% for train and 30% for test\n'))
    # except:
    #     split_ratio = 0.3  
    split_ratio = float(split_ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, random_state = 6) 

    print("Xtrain:", X_train.shape)
    print("Ytrain, Ytrain with mino class: ", y_train.shape, y_train.values.tolist().count(1))
    return X_train, X_test, y_train, y_test    

#SMOTE on Train Data
def smote_data(X_train, X_test, y_train, y_test):
    sm = SMOTE(random_state = 7, sampling_strategy=0.3) 
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 
    print("Xtrain smoted:", X_train_res.shape)
    print("Ytrain smoted, Ytrain smoted with mino class: ", y_train_res.shape, y_train_res.tolist().count(1))
    return X_train_res, X_test, y_train_res, y_test

@app.route('/')
def file_input():
    return render_template('dynamic_input.html')

@app.route('/results', methods = ['GET', 'POST'])
def results():
    global df
    if request.method == 'GET':
        return redirect(url_for('/'))
    else:
        values = request.form.getlist('input_text[]')
        print(values)
        df = concat_files(values)
        print("NULL COUNT")
        null_count = [(col, df[col].isnull().sum()) for col in df.columns]
        return render_template('dynamic_input_results.html',
                               values = df.columns.tolist(), null_count=null_count, 
                               head=df.head().to_html(), info=df.dtypes)


@app.route('/results_drop', methods = ['GET', 'POST'])
def results_drop():
    global df, X, rf_model, xgb_model, svm_model, gnb_model, mnb_model, lr_model, logr_model
    if request.method == 'GET':
        return redirect(url_for('/'))
    else:
        cols_to_drop = request.form.getlist('input_text[]')
        target = request.form.get('target')
        split_ratio = request.form.get('split_ratio')
        smote_ratio = request.form.get('smote_ratio')
        print(cols_to_drop)
        print("Target:", target)
        # return render_template('dynamic_input_results.html',
        #                        values = cols.tolist()) 
    X,Y = def_X_y(cols_to_drop, target)
    X_train, X_test, y_train, y_test = split(X,Y, split_ratio)
    if float(smote_ratio)>0:
        X_train, X_test, y_train, y_test = smote_data(X_train, X_test, y_train, y_test)
    html_res, rf_model, lgbm_model, xgb_model, svm_model, gnb_model, mnb_model, lr_model, logr_model = execut(X_train, X_test, y_train, y_test)
    return render_template('train_results.html', html_res=html_res)

@app.route('/predict', methods=['GET', 'POST'])    
def predict():
    global X, rf_model, lgbm_model, xgb_model, svm_model, gnb_model, mnb_model, lr_model, logr_model
    results_df = "ERROR"
    if request.method == 'GET':
        return redirect(url_for('/'))
    else:
        pred_file = request.form.get('prediction_file')        
        print(pred_file)
        pred_df = read_file(pred_file)
        if ((set(pred_df.columns) - set(X.columns))==set()) and ((set(X.columns) - set(pred_df.columns))==set()):
            pred_df.fillna(9, inplace=True)
            models = [rf_model, xgb_model, svm_model, gnb_model, mnb_model, lr_model, logr_model]
            models_names = ['rf_model','xgb_model','svm_model','gnb_model','mnb_model','lr_model','logr_model']
            results_df = pd.DataFrame(columns=models_names)
            print("MODELS\n", models)
            for model,name in zip(models,models_names):
                print(type(model))
                results_df[name] = model.predict(pred_df)
        else:
            return "Column mismatch"    
    return results_df.to_html(classes='data')


def calc_rocauc(Y_test):
    ns_probs = [0 for _ in range(len(Y_test))]
    ns_auc = roc_auc_score(Y_test, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

if __name__ == '__main__':
    app.run(debug=True, port=5500)    
