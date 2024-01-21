# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:23:50 2024


import pandas as pd
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet

def read_file(name):
    """
    Read data from a CSV file, skipping the first 4 rows.
    Return the original data and its transpose.
    """
    data = pd.read_csv(name, skiprows=4)
    return data, data.T

def error_ranges(x, func, param, sigma):
    """
    Calculates the higher and decrease limits for the function, parameters, and sigmas.
    """
    import itertools as iter

    lower = func(x, *param)
    upper = lower

    uplow = [(p - s, p + s) for p, s in zip(param, sigma)]
    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper

def logistic(t, n0, g, t0):
    """
    Calculates the logistic feature with scale component n0 and boom charge g.
    """
    return n0 / (1 + np.exp(-g * (t - t0)))

def plot_data(data_for_fitting, name, feature, xlabel, ylabel):
    """
    Plot the data, fitted curve, and forecast with confidence intervals.
    """
    year = np.arange(1963, 2021)
    popt, covar = opt.curve_fit(logistic, data_for_fitting['Year'], data_for_fitting[feature], p0=(2e9, 0.05, 1990.0))
    data_for_fitting["fit"] = logistic(data_for_fitting["Year"], *popt)
    sigma = np.sqrt(np.diag(covar))
    forecast = logistic(year, *popt)
    low, up = error_ranges(year, logistic, popt, sigma)

    data_for_fitting.plot("Year", [feature, "fit"])
    plt.title(f"{name} {feature.capitalize()} Fitting")
    plt.ylabel(ylabel)
    plt.show()

    plt.figure()
    plt.plot(data_for_fitting["Year"], data_for_fitting[feature], label=feature)
    plt.title(f"{name} {feature.capitalize()} Fitting")
    plt.plot(year, forecast, label="Forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def fitting(data, name):
    """
    This function will separate facts symptoms and then plot diagrams exhibiting the top information.
    It will also exhibit self-assurance of statistics and error top and decrease restrict.
    """
    # Fit exponential growth
    data_for_fitting = pd.DataFrame()
    year = np.arange(1963, 2020)

    print(year)

    in_data = data[data["Country Name"] == name]

    if not in_data.empty:
        in_data_fores_area = in_data[in_data["Indicator Code"] == "EN.ATM.CO2E.LF.KT"]
        in_data_fores_urban = in_data[in_data["Indicator Code"] == "SP.URB.TOTL"]
        in_data_fores_arable_land = in_data[in_data["Indicator Code"] == "EN.ATM.CO2E.LF.KT"]

        in_data_fores_area = in_data_fores_area.drop(["Country Name", "Indicator Name", "Country Code", "Indicator Code"], axis=1).T
        in_data_fores_urban = in_data_fores_urban.drop(["Country Name", "Indicator Name", "Country Code", "Indicator Code"], axis=1).T
        in_data_fores_arable_land = in_data_fores_arable_land.drop(["Country Name", "Indicator Name", "Country Code", "Indicator Code"], axis=1).T

        in_data_fores_area = in_data_fores_area.dropna()
        in_data_fores_urban = in_data_fores_urban.dropna()
        in_data_fores_arable_land = in_data_fores_arable_land.dropna()

        data_for_fitting['co2'] = in_data_fores_area
        data_for_fitting['urban'] = in_data_fores_urban
        data_for_fitting['arable'] = in_data_fores_arable_land
        data_for_fitting['Year'] = pd.to_numeric(year)

        plot_data(data_for_fitting, name, 'urban', 'Year', 'Urban Population')
        plot_data(data_for_fitting, name, 'co2', 'Year', 'CO2 Emissions')
        plot_data(data_for_fitting, name, 'arable', 'Year', 'Arable Land')

        return data_for_fitting

    else:
        print(f"No data available for {name}.")
        return pd.DataFrame()

def k_means_clustering(data, xlabel, ylabel):
    """
    Exhibit the assessment of specific capacity K-means cluster.
    """
    df_ex = data[["co2", "urban"]].copy()
    max_val = df_ex.max()
    min_val = df_ex.min()
    df_ex = (df_ex - min_val) / (max_val - min_val)

    ncluster = 3
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_ex)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    print(cen)
    print(skmet.silhouette_score(df_ex, labels))

    plt.figure(figsize=(10.0, 10.0))
    col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    for l in range(ncluster):
        plt.plot(df_ex[labels==l]["co2"], df_ex[labels==l]["urban"], "o", markersize=3, color=col[l])

    for ic in range(ncluster):
        xc, yc = cen[ic, :]
        plt.plot(xc, yc, "dk", markersize=10, label=f"Cluster {ic}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title("Indicate Cluster of Data")
    plt.show()

if __name__ == "__main__":
    data, transposed_data = read_file("/content/Dataset.csv")

    filter_china = fitting(data, "China")
    k_means_clustering(filter_china, "co2", "Arable")

    filter_United_States = fitting(data, "United States")
    k_means_clustering(filter_United_States, "co2", "Arable")

