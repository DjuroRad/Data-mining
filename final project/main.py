# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

#PERFORM FEATURE SELECTION ACCORDING TO THE THRESHOLD
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr('pearson')

    for i_ in range(len(corr_matrix.columns)):

        for j_ in range(i_):
            if abs(corr_matrix.iloc[i_, j_]) > threshold:#keep it if it is above the threshold!
                colname = corr_matrix.columns[i_]
                col_corr.add(colname)

    return col_corr

def linearRegression(show_detailed):
    from sklearn.datasets import load_boston
    # Importing the dataset
    dataset = pd.read_csv('diabetes.csv')
    #making sure that binary string for gender are converted to binary 0/1 value
    # gender = {'male': 1, 'female':0 }
    # dataset.Gender[dataset.Gender == 'male'] = dataset.Gender[dataset.Gender == 'male'] = 1
    # dataset.Gender[dataset.Gender == 'female'] = 0
    # dataset['Gender'] = (dataset['Gender'] == 'Male').astype(int)
    dataset_preprocessed = dataset
    dataset_preprocessed = dataset_preprocessed.drop('Outcome', axis=1)

    #feature selection using correlation in here
    # fig_dims = ( len(dataset_preprocessed), len(dataset_preprocessed.columns))
    # fig, ax = plt.subplots(figsize=fig_dims)
    # sn.heatmap()
    #
    # print(dataset_preprocessed)
    # print(dataset_preprocessed.feature_names)
    # exit()


    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
    y = dataset.iloc[:, 8].values

    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    if(show_detailed):
        plt.figure(figsize=(12,10))
        cor = dataset_preprocessed.corr(method='pearson')
        sn.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        plt.show()

    corr_features = correlation(dataset_preprocessed, 0.35)

    for col in dataset_preprocessed.columns:
        if not(col in corr_features):
            dataset_preprocessed = dataset_preprocessed.drop([col], axis=1)

    # print(dataset_preprocessed)
    # print(corr_features)

    # Since value of some of the data is very big and of the others is small we should perform feature scaling
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Logistic Regression to the Training set
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)



    #plotting the ROC - curve after this is performed properly
    y_pred_roc = classifier.decision_function(X_test)

    fpr, tpr, th = roc_curve(y_test, y_pred_roc)
    auc_ = auc(fpr, tpr)

    if( show_detailed ):
        print("\n\n\nLOGISTIC REGRESSION")
        print("Confusion matrix")
        print(cm)
        print("Following results were recorded")
        print(classification_report(y_test, y_pred))
        print("Accuracy score is: ", accuracy_score(y_test, y_pred))

        plt.figure(figsize=(5,5), dpi=100)
        plt.plot(fpr, tpr, linestyle = '-', label='LOGISTIC REGRESSION ( auc = %0.3f)' %auc_)
        plt.xlabel('FPR')
        plt.ylabel("TPR")
        plt.legend()
        plt.show()

    return fpr, tpr, th, auc_

def kNearestNeighbour(show_detailed):

    dataset = pd.read_csv('diabetes.csv')
    dataset_preprocessed = dataset
    dataset_preprocessed = dataset_preprocessed.drop('Outcome', axis=1)

    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
    y = dataset.iloc[:, 8].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    if(show_detailed):
        plt.figure(figsize=(12,10))
        cor = dataset_preprocessed.corr(method='pearson')
        sn.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        plt.show()

    corr_features = correlation(dataset_preprocessed, 0.35)

    for col in dataset_preprocessed.columns:
        if not(col in corr_features):
            dataset_preprocessed = dataset_preprocessed.drop([col], axis=1)

    # print(dataset_preprocessed)
    # print(corr_features)

    # Since value of some of the data is very big and of the others is small we should perform feature scaling
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Logistic Regression to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)  # defult neighboru = 5 iyi

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    #plotting the ROC - curve after this is performed properly
    y_pred_roc = classifier.predict_proba(X_test)

    fpr, tpr, th = roc_curve(y_test, y_pred_roc[:,1])
    auc_ = auc(fpr, tpr)

    if( show_detailed ):
        print("\n\n\nKNEARESTNEIGHBOUR")
        print("Confusion matrix")
        print(cm)
        print("Following results were recorded")
        print(classification_report(y_test, y_pred))
        print("Accuracy score is: ", accuracy_score(y_test, y_pred))

        plt.figure(figsize=(5,5), dpi=100)
        plt.plot(fpr, tpr, linestyle = '-', label='LOGISTIC REGRESSION ( auc = %0.3f)' %auc_)
        plt.xlabel('FPR')
        plt.ylabel("TPR")
        plt.legend()
        plt.show()

    return fpr, tpr, th, auc_

def SVM(show_detailed):

    dataset = pd.read_csv('diabetes.csv')
    dataset_preprocessed = dataset
    dataset_preprocessed = dataset_preprocessed.drop('Outcome', axis=1)

    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
    y = dataset.iloc[:, 8].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    if(show_detailed):
        plt.figure(figsize=(12,10))
        cor = dataset_preprocessed.corr(method='pearson')
        sn.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        plt.show()

    corr_features = correlation(dataset_preprocessed, 0.35)

    for col in dataset_preprocessed.columns:
        if not(col in corr_features):
            dataset_preprocessed = dataset_preprocessed.drop([col], axis=1)

    # print(dataset_preprocessed)
    # print(corr_features)

    # Since value of some of the data is very big and of the others is small we should perform feature scaling
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Logistic Regression to the Training set
    classifier = SVC(kernel = 'linear', random_state = 0)

    # metrics.plot_roc_curve(classifier, X_test, y_test)
    # plt.show()
    # exit()

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)


    y_pred_roc = classifier.decision_function(X_test)

    #plotting the ROC - curve after this is performed properly
    # y_pred_roc = classifier.decision_function(X_test)

    fpr, tpr, th = roc_curve(y_test, y_pred_roc)
    auc_ = auc(fpr, tpr)

    if( show_detailed ):
        print("\n\n\nSVM")
        print("Confusion matrix")
        print(cm)
        print("Following results were recorded")
        print(classification_report(y_test, y_pred))
        print("Accuracy score is: ", accuracy_score(y_test, y_pred))

        plt.figure(figsize=(5,5), dpi=100)
        plt.plot(fpr, tpr, linestyle = '-', label='SVM( auc = %0.3f)' %auc_)
        plt.xlabel('FPR')
        plt.ylabel("TPR")
        plt.legend()
        plt.show()

    return fpr, tpr, th, auc_



def SVM_KERNEL(show_detailed):

    dataset = pd.read_csv('diabetes.csv')
    dataset_preprocessed = dataset
    dataset_preprocessed = dataset_preprocessed.drop('Outcome', axis=1)

    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
    y = dataset.iloc[:, 8].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    if(show_detailed):
        plt.figure(figsize=(12,10))
        cor = dataset_preprocessed.corr(method='pearson')
        sn.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        plt.show()

    corr_features = correlation(dataset_preprocessed, 0.35)

    for col in dataset_preprocessed.columns:
        if not(col in corr_features):
            dataset_preprocessed = dataset_preprocessed.drop([col], axis=1)

    # print(dataset_preprocessed)
    # print(corr_features)

    # Since value of some of the data is very big and of the others is small we should perform feature scaling
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Logistic Regression to the Training set
    classifier = SVC(kernel = 'rbf', random_state = 0)

    # metrics.plot_roc_curve(classifier, X_test, y_test)
    # plt.show()
    # exit()

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)


    y_pred_roc = classifier.decision_function(X_test)

    #plotting the ROC - curve after this is performed properly
    # y_pred_roc = classifier.decision_function(X_test)

    fpr, tpr, th = roc_curve(y_test, y_pred_roc)
    auc_ = auc(fpr, tpr)

    if( show_detailed ):
        print("\n\n\nSVM KERNEL")
        print("Confusion matrix")
        print(cm)
        print("Following results were recorded")
        print(classification_report(y_test, y_pred))
        print("Accuracy score is: ", accuracy_score(y_test, y_pred))

        plt.figure(figsize=(5,5), dpi=100)
        plt.plot(fpr, tpr, linestyle = '-', label='SVM KERNEL( auc = %0.3f)' %auc_)
        plt.xlabel('FPR')
        plt.ylabel("TPR")
        plt.legend()
        plt.show()

    return fpr, tpr, th, auc_



def NaiveBayes(show_detailed):

    dataset = pd.read_csv('diabetes.csv')
    dataset_preprocessed = dataset
    dataset_preprocessed = dataset_preprocessed.drop('Outcome', axis=1)

    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
    y = dataset.iloc[:, 8].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    if(show_detailed):
        plt.figure(figsize=(12,10))
        cor = dataset_preprocessed.corr(method='pearson')
        sn.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        plt.show()

    corr_features = correlation(dataset_preprocessed, 0.35)

    for col in dataset_preprocessed.columns:
        if not(col in corr_features):
            dataset_preprocessed = dataset_preprocessed.drop([col], axis=1)

    # print(dataset_preprocessed)
    # print(corr_features)

    # Since value of some of the data is very big and of the others is small we should perform feature scaling
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Logistic Regression to the Training set
    classifier = GaussianNB()

    # metrics.plot_roc_curve(classifier, X_test, y_test)
    # plt.show()
    # exit()

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)


    # y_pred_roc = classifier.predict_proba(X_test)

    #plotting the ROC - curve after this is performed properly
    y_pred_roc = classifier.predict_proba(X_test)

    fpr, tpr, th = roc_curve(y_test, y_pred_roc[:, 1])
    auc_ = auc(fpr, tpr)

    if( show_detailed ):
        print("\n\n\nNAIVE BAYES")
        print("Confusion matrix")
        print(cm)
        print("Following results were recorded")
        print(classification_report(y_test, y_pred))
        print("Accuracy score is: ", accuracy_score(y_test, y_pred))

        plt.figure(figsize=(5,5), dpi=100)
        plt.plot(fpr, tpr, linestyle = '-', label='NaiveBayes')

    return fpr, tpr, th, auc_


def DecisionTree(show_detailed):

    dataset = pd.read_csv('diabetes.csv')
    dataset_preprocessed = dataset
    dataset_preprocessed = dataset_preprocessed.drop('Outcome', axis=1)

    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
    y = dataset.iloc[:, 8].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    if(show_detailed):
        plt.figure(figsize=(12,10))
        cor = dataset_preprocessed.corr(method='pearson')
        sn.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        plt.show()

    corr_features = correlation(dataset_preprocessed, 0.35)

    for col in dataset_preprocessed.columns:
        if not(col in corr_features):
            dataset_preprocessed = dataset_preprocessed.drop([col], axis=1)

    # print(dataset_preprocessed)
    # print(corr_features)

    # Since value of some of the data is very big and of the others is small we should perform feature scaling
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.tree import DecisionTreeClassifier
    # Fitting Logistic Regression to the Training set
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    # metrics.plot_roc_curve(classifier, X_test, y_test)
    # plt.show()
    # exit()

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)


    # y_pred_roc = classifier.predict_proba(X_test)

    #plotting the ROC - curve after this is performed properly
    y_pred_roc = classifier.predict_proba(X_test)

    fpr, tpr, th = roc_curve(y_test, y_pred_roc[:, 1])
    auc_ = auc(fpr, tpr)

    if( show_detailed ):
        print("\n\n\nDECISIONTREE")
        print("Confusion matrix")
        print(cm)
        print("Following results were recorded")
        print(classification_report(y_test, y_pred))
        print("Accuracy score is: ", accuracy_score(y_test, y_pred))

        plt.figure(figsize=(5,5), dpi=100)
        plt.plot(fpr, tpr, linestyle = '-', label='Decision Tree')
        plt.show()

    return fpr, tpr, th, auc_


def RandomForest(show_detailed):

    dataset = pd.read_csv('diabetes.csv')
    dataset_preprocessed = dataset
    dataset_preprocessed = dataset_preprocessed.drop('Outcome', axis=1)

    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
    y = dataset.iloc[:, 8].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    if(show_detailed):
        plt.figure(figsize=(12,10))
        cor = dataset_preprocessed.corr(method='pearson')
        sn.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        plt.show()

    corr_features = correlation(dataset_preprocessed, 0.35)

    for col in dataset_preprocessed.columns:
        if not(col in corr_features):
            dataset_preprocessed = dataset_preprocessed.drop([col], axis=1)

    # print(dataset_preprocessed)
    # print(corr_features)

    # Since value of some of the data is very big and of the others is small we should perform feature scaling
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.tree import DecisionTreeClassifier
    # Fitting Logistic Regression to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)    # metrics.plot_roc_curve(classifier, X_test, y_test)
    # plt.show()
    # exit()

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)


    # y_pred_roc = classifier.predict_proba(X_test)

    #plotting the ROC - curve after this is performed properly
    y_pred_roc = classifier.predict_proba(X_test)

    fpr, tpr, th = roc_curve(y_test, y_pred_roc[:, 1])
    auc_ = auc(fpr, tpr)

    if( show_detailed ):
        print("\n\n\nRandom Forest")
        print("Confusion matrix")
        print(cm)
        print("Following results were recorded")
        print(classification_report(y_test, y_pred))
        print("Accuracy score is: ", accuracy_score(y_test, y_pred))

        plt.figure(figsize=(5,5), dpi=100)
        plt.plot(fpr, tpr, linestyle = '-', label='Random Forest')
        plt.show()

    return fpr, tpr, th, auc_



#PERCEPTRON SELF-IMPLEMENTATION



class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i>0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                dot_prod = np.dot(x_i, self.weights)
                linear_output = dot_prod + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr = (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)



#PERCEPTRON IMPLEMENTATION

def PerceptronTest(show_detailed):

    dataset = pd.read_csv('diabetes.csv')
    dataset_preprocessed = dataset
    dataset_preprocessed = dataset_preprocessed.drop('Outcome', axis=1)

    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
    y = dataset.iloc[:, 8].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    if(show_detailed):
        plt.figure(figsize=(12,10))
        cor = dataset_preprocessed.corr(method='pearson')
        sn.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        plt.show()

    corr_features = correlation(dataset_preprocessed, 0.35)

    for col in dataset_preprocessed.columns:
        if not(col in corr_features):
            dataset_preprocessed = dataset_preprocessed.drop([col], axis=1)

    # print(dataset_preprocessed)
    # print(corr_features)

    # Since value of some of the data is very big and of the others is small we should perform feature scaling
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.tree import DecisionTreeClassifier
    # Fitting Logistic Regression to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = Perceptron(learning_rate = 0.01, n_iters = 1000)    # metrics.plot_roc_curve(classifier, X_test, y_test)
    # plt.show()

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)


    # y_pred_roc = classifier.predict_proba(X_test)

    #plotting the ROC - curve after this is performed properly
    # y_pred_roc = classifier.predict_proba(X_test)

    fpr, tpr, th = roc_curve(y_test, y_pred)
    auc_ = auc(fpr, tpr)

    if( show_detailed ):
        print("\n\n\nPERCEPTRON")
        print("Confusion matrix")
        print(cm)
        print("Following results were recorded")
        print(classification_report(y_test, y_pred))
        print("Accuracy score is: ", accuracy_score(y_test, y_pred))

        plt.figure(figsize=(5,5), dpi=100)
        plt.plot(fpr, tpr, linestyle = '-', label='Perceptron')
        plt.show()

        #graph perceprtron in order to understand its linearity
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.scatter(X_train[:,0], X_train[:,1], marker='o', c=y_train)

        x01 = np.amin(X_train[:,0])
        x02 = np.amax(X_train[:,0])

        x11 = (-classifier.weights[0]*x01 - classifier.bias)/classifier.weights[1]
        x12 = (-classifier.weights[0]*x02 - classifier.bias)/classifier.weights[1]

        ax.plot([x01, x02], [x11,x12], 'k')

        ymin = np.amin(X_train[:,1])
        ymax = np.amax(X_train[:,1])

        ax.set_ylim([ymin-3,ymax+3])

        plt.show()
    return fpr, tpr, th, auc_



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fpr_lr, tpr_lr, t, a = linearRegression( False )
    fpr_kn, tpr_kn, t_kn, a_kn = kNearestNeighbour( False )
    fpr_svm, tpr_svm, t_svm, a_svm = SVM(False)
    fpr_svm_k, tpr_svm_k, t_svm_k, a_svm_k = SVM_KERNEL( False )
    fpr_nb, tpr_nb, t_nb, a_nb = NaiveBayes( False )
    fpr_dt, tpr_dt, t_dt, a_dt = DecisionTree( False )
    fpr_rf, tpr_rf, t_rf, a_rf = RandomForest( False )
    PerceptronTest(True)
    exit()

    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(fpr_lr, tpr_lr, linestyle='-', label='LOGISTIC REGRESSION ( auc = %0.3f)' % a)
    plt.plot(fpr_kn, tpr_kn, linestyle='-', label='KNEIGHBOUR REGRESSION ( auc = %0.3f)' % a_kn)
    plt.plot(fpr_svm, tpr_svm, linestyle='-', label='SVM ( auc = %0.3f)' % a_svm)
    plt.plot(fpr_nb, tpr_nb, linestyle='-', label='NAIVE BAYERS ( auc = %0.3f)' % a_nb)
    plt.plot(fpr_dt, tpr_dt, linestyle='-', label='DECISION TREE BAYERS ( auc = %0.3f)' % a_dt)
    plt.plot(fpr_rf, tpr_rf, linestyle='-', label='RANDOM FOREST ( auc = %0.3f)' % a_rf)

    plt.xlabel('FPR')
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
