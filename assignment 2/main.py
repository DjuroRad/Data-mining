import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from clustviz.chameleon.chameleon import cluster
from mlxtend.preprocessing import TransactionEncoder
from fpgrowth_py import fpgrowth
from matplotlib import pyplot as plt

def dbscan_alg():
    # print("DBSCAN IS WORKING HERE")
    df = pd.read_csv("salary_dataset.csv");
    # print(df[['Age','Salary']])
    plt.scatter(df['Age'], df['Salary'])



    start = time.time()
    clustering = DBSCAN(eps=5000, min_samples=2).fit(df[['Age','Salary']])
    end = time.time()
    print(f"Runtime of DBSCAN algorithm on D1 is: {end-start}")

    # print(clustering.labels_)
    df['cluster'] = clustering.labels_

    # print(df)
    # print(df.head(30))
    df1 = df[df.cluster==0]
    df2 = df[df.cluster==1]
    df3 = df[df.cluster==2]
    df4 = df[df.cluster==3]
    df5 = df[df.cluster==4]
    df6 = df[df.cluster==5]
    df7 = df[df.cluster==6]
    dfnoise = df[df.cluster==-1]

    plt.scatter(df1.Age, df1['Salary'], color = 'green')
    plt.scatter(df2.Age, df2['Salary'], color = 'blue')
    plt.scatter(df3.Age, df3['Salary'], color = 'yellow')
    plt.scatter(df4.Age, df4['Salary'], color = 'orange')
    plt.scatter(df5.Age, df5['Salary'], color = 'teal')
    plt.scatter(df6.Age, df6['Salary'], color = 'purple')
    plt.scatter(df7.Age, df7['Salary'], color = 'gray')
    plt.scatter(dfnoise.Age, dfnoise['Salary'], color = 'red')

    plt.xlabel('Age')
    plt.ylabel('Salary')

    return df

def dbscan_2():
    start = time.time()
    df = cleanAndNormalizeDS2()

    clustering = DBSCAN(eps=0.24, min_samples=2).fit(df)
    # km = KMeans(n_clusters=71)
    # y_predicted = km.fit_predict(df)
    end = time.time()

    print(f"Execution time of DBSCAN algorithm with DB2 is {end - start}")

    df['cluster'] = clustering.labels_

    df2 = df
    del df2['cluster']
    # print(km.labels_)
    score = silhouette_score(df2, clustering.labels_)
    print(f"Silhouette score for DBSCAN algorithm with DS2 is {score}")

def dbscan_test():
    df = dbscan_alg()
    score = silhouette_score(df[['Age', 'Salary']], df['cluster'])
    print(f"Silhouette score for DB-scan is {score}")

    plt.plot()
    plt.show()

def kmeans_alg():
    df = pd.read_csv("salary_dataset.csv");
    # print(df.head(100))
    plt.scatter(df['Age'], df['Salary'])
    km = KMeans(n_clusters=4)

    #scale the data here since it should be 0-1 ideally
    scaled = NormalizeData(df[['Age', 'Salary']])

    start = time.time()
    y_predicted = km.fit_predict(scaled)
    end = time.time()
    print(f"Execution time of K-means algorithm is {end-start}")
    # print(y_predicted)
    df['cluster'] = y_predicted

    # print(df.head(30))
    df1 = df[df.cluster==0]
    df2 = df[df.cluster==1]
    df3 = df[df.cluster==2]
    df4 = df[df.cluster==3]

    plt.scatter(df1.Age, df1['Salary'], color = 'green')
    plt.scatter(df2.Age, df2['Salary'], color = 'blue')
    plt.scatter(df3.Age, df3['Salary'], color = 'yellow')
    plt.scatter(df4.Age, df4['Salary'], color = 'orange')

    plt.xlabel('Age')
    plt.ylabel('Salary')
    return df

def k_Means_Test():
    df = kmeans_alg()
    score = silhouette_score(df[['Age', 'Salary']], df['cluster'])
    print(f"Silhouette score for K-means is {score}")

    plt.plot()
    plt.show()

def chameleon(df):
    # df = pd.read_csv("salary_dataset.csv");
    # print(df.head(100))
    start = time.time()
    plt.scatter(df['Age'], df['Salary'])
    # km = KMeans(n_clusters=4)
    cham = cluster(df, k = 2, knn = 4, m = 6, alpha = 2, plot = False, verbose0=False, verbose1=False, verbose2=False)
    # df = cham[0].cluster
    df['cluster'] = cham[0].cluster
    # print(df.head(30))
    df1 = df[df.cluster == 1]
    df2 = df[df.cluster == 2]
    df3 = df[df.cluster == 3]
    df4 = df[df.cluster == 4]

    plt.scatter(df1.Age, df1['Salary'], color='green')
    plt.scatter(df2.Age, df2['Salary'], color='blue')
    plt.scatter(df3.Age, df3['Salary'], color='yellow')
    plt.scatter(df4.Age, df4['Salary'], color='orange')

    plt.xlabel('Age')
    plt.ylabel('Salary')

    end = time.time()
    print(f"Runtime of Chameleon algorithm on D1 is: {end-start}")

    score = silhouette_score(df[['Age', 'Salary']], df['cluster'])
    print(f"Silhouette score for Chameleon algorithm is {score}")

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def cleanAndNormalizeDS2():
    df = pd.read_csv("CarPrice_Assignment.csv");
    # print(df.head(100))
    del df['CarName']
    del df['fueltype']
    del df['aspiration']
    del df['doornumber']
    del df['carbody']
    del df['drivewheel']
    del df['enginelocation']
    del df['enginetype']
    del df['fuelsystem']
    del df['cylindernumber']
    for i in df:
        df[i] = NormalizeData(df[i])

    return df


def kmeans_alg2():
    start = time.time()
    df = cleanAndNormalizeDS2()
    km = KMeans(n_clusters=71)
    y_predicted = km.fit_predict(df)
    end = time.time()

    print(f"Execution time of K-means algorithm with DB2 is {end-start}")

    df['cluster'] = y_predicted

    df2 = df
    del df2['cluster']
    # print(km.labels_)
    score = silhouette_score(df2, km.labels_)
    print(f"Silhouette score for K-means algorithm with DS2 is {score}")


def chameleon2():
    start = time.time()
    df = cleanAndNormalizeDS2()

    cham = cluster(df, k=2, knn=4, m=6, alpha=2, plot=False, verbose0=False, verbose1=False, verbose2=False)
    # df = cham[0].cluster
    # df['cluster'] = cham[0].cluster
    # clustering = DBSCAN(eps=0.24, min_samples=2).fit(df)
    # km = KMeans(n_clusters=71)
    # y_predicted = km.fit_predict(df)
    end = time.time()

    print(f"Execution time of Chameleon algorithm with DB2 is {end - start}")

    # df2 = df
    # del df2['cluster']
    # print(km.labels_)
    # print(cham[0].cluster)
    score = silhouette_score(df, cham[0].cluster)
    print(f"Silhouette score for Chameleon algorithm with DS2 is {score}")

def fpgrowthtest():

    # Open file
    fileHandler = open("groceries.csv", "r")
    # Get list of all lines in file
    listOfLines = fileHandler.readlines()
    # Close file
    fileHandler.close()

    dataset = []
    for line in listOfLines:
        line = line.replace("\n", "")
        dataset.append(line.split(','))

    dataset = dataset[:10]
    # print(dataset)
    #
    # itemSetList = [['eggs', 'bacon', 'soup'],
    #                ['eggs', 'bacon', 'apple'],
    #                ['soup', 'bacon', 'banana']]
    # print (itemSetList)
    start = time.time()
    freqItemSet, rules = fpgrowth(dataset, minSupRatio=0.1, minConf=0.1)
    end = time.time()
    print(f"Runtime of FPgrowth  algorithm for a grocery dataset is: {end - start}")
    print("Frequent itemset")
    print(freqItemSet)
    print("Rules")
    print(rules)

def main():
    k_Means_Test()
    dbscan_test()
    df = pd.read_csv("salary_dataset.csv");
    # print(df[['Age', 'Salary']])
    chameleon(df[['Age', 'Salary']])

    plt.plot()
    plt.show()
    kmeans_alg2()
    dbscan_2()
    chameleon2()
    fpgrowthtest()

# Using the special variable
# __name__
if __name__=="__main__":
    main()