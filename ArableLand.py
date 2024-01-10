import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
from scipy.optimize import curve_fit
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


def read_file(file_path):
    """
    Reads a CSV file and returns a DataFrame.

    Parameters:
    file_path (str): The path to the CSV file to be read.

    Returns:
    DataFrame: A DataFrame containing the data read from the CSV file.
    """
    data = pd.read_csv(file_path)
    return data


def calculate_silhouette(xy, n_clusters):
    """
    Calculates the silhouette score for a given number of clusters in a dataset.

    Parameters:
    xy (array): The data for clustering.
    n_clusters (int): The number of clusters to form.

    Returns:
    float: The silhouette score
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = silhouette_score(xy, labels)
    return score


def polynomial(x, a, b, c, d):
    """
    Polynomial model function for calculating the value of a polynomial at x.

    Parameters:
    x (number or array): The value(s) at which the polynomial is evaluated.
    a, b, c, d (number): Coefficients of the polynomial.

    Returns:
    number or array: The value of the polynomial at x.
    """
    return a * x**3 + b * x**2 + c * x + d


def calculate_error(x, func, params, covariance):
    """
    Calculates error propagation for a given function and its parameters.

    Parameters:
    x (number or array): The input value(s) at which the error is evaluated.
    func (function): The function for which the error is calculated.
    params (list or array): Parameters of the function.
    covariance (array): The covariance matrix of the parameters.

    Returns:
    numpy.ndarray: The calculated error values corresponding to each x.
    """
    var = np.zeros_like(x)
    for i in range(len(params)):
        deriv1 = calculate_derivative(x, func, params, i)
        for j in range(len(params)):
            deriv2 = calculate_derivative(x, func, params, j)
            var += deriv1 * deriv2 * covariance[i, j]
    return np.sqrt(var)


def calculate_derivative(x, func, params, index):
    """
    Calculates the derivative of a function wrt to one of its parameters.

    Parameters:
    x (number or array): The value(s) at which the derivative is calculated.
    func (function): The function for which the derivative is calculated.
    params (list or array): Parameters of the function.
    index (int): The index of the parameter wrt to which the deriv is calculated.

    Returns:
    numpy.ndarray or number: The deriv of the function wrt to the parameter.
    """
    scale = 1e-6
    delta = np.zeros_like(params)
    delta[index] = scale * abs(params[index])
    up_lim = params + delta
    low_lim = params - delta
    diff = 0.5 * (func(x, *up_lim) - func(x, *low_lim))
    return diff / (scale * abs(params[index]))


# Read and analyze the data
inf_data = read_file("ArableLand.csv")
if inf_data is not None:
    print(inf_data.describe())

inf_data = inf_data[(inf_data["1960"].notna()) & 
                                 (inf_data["2020"].notna())]
inf_data.reset_index(drop=True, inplace=True)

growth = inf_data[["Country Name", "1960"]].copy()
growth["Growth"] = 100.0 / 20.0 * (inf_data["2020"] - 
                    inf_data["1960"]) / inf_data["1960"]
warnings.filterwarnings("ignore", category=UserWarning)

print(growth.describe())
print()
print(growth.dtypes)

#scatter plot for original data
plt.figure(figsize=(8, 8))
plt.scatter(growth["1960"], growth["Growth"])
plt.xlabel("Arable land(hect per person),1960")
plt.ylabel("Growth/year in %")
plt.show()

# normalization of data
scaler = RobustScaler()
df_clust = growth[["1960", "Growth"]]
scaler.fit(df_clust)
norm_data = scaler.transform(df_clust)

# scatter plot for normalized data
plt.figure(figsize=(8, 8))
plt.scatter(norm_data[:, 0], norm_data[:, 1])
plt.xlabel("Normalized Arable land(hect per person),1960")
plt.ylabel("Normalized Growth per year [%]")
plt.show()

# Fetching silhoutte score
for cluster_count in range(2, 11):
    score = calculate_silhouette(norm_data, cluster_count)
    score_message = f"The silhouette score for {cluster_count:3d} clusters is"
    print(f"{score_message} {score:7.4f}")


# Clustering with KMeans
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
kmeans.fit(norm_data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
cen = scaler.inverse_transform(centers)

# Initial Scatter Plot with clusters
plt.figure(figsize=(8, 8))
scatter = plt.scatter(growth["1960"], growth["Growth"], c=labels, 
                      s=10, marker="o", cmap=cm.rainbow, label='Data Points')
cen_scatter = plt.scatter(cen[:, 0], cen[:, 1],
                              s=45, c="k", marker="d", label='Cluster Centers')
plt.xlabel("Arable land(hect per person),1960")
plt.ylabel("Growth/year in %")


# Adding legend
plt.legend(handles=[scatter, cen_scatter], loc='upper right')
plt.show()

print(cen)

# Fetching subset of original data
growth_2 = growth[labels == 0].copy()
print(growth_2.describe())
#Normalising the data
df_clust2 = growth_2[["1960", "Growth"]]
scaler.fit(df_clust2)
norm_dt2 = scaler.transform(df_clust2)

plt.figure(figsize=(8, 8))
plt.scatter(norm_dt2[:, 0], norm_dt2[:, 1])
plt.xlabel("Arable land(hect per person),1960")
plt.ylabel("Growth/year in %")
plt.show()

# clustering within subset
kmeans_subset = cluster.KMeans(n_clusters=3, n_init=20)
kmeans_subset.fit(norm_dt2)
subset_labels = kmeans_subset.labels_
subset_centers = kmeans_subset.cluster_centers_

#Backscaling the cluster centers
SS_cen = scaler.inverse_transform(subset_centers)

# Second Scatter Plot (Within Subset):
plt.figure(figsize=(8.0, 8.0))
scatter_SS = plt.scatter(growth_2["1960"], growth_2["Growth"], 
                             c=subset_labels, s=10, marker="o", 
                             cmap=cm.rainbow,
                             label='Data Points')
SS_cen = plt.scatter(SS_cen[:, 0], 
                                     SS_cen[:, 1],
                                     s=45, c="k", marker="d", 
                                     label='Cluster Centers')
plt.xlabel("Arable land(hect per person),1960")
plt.ylabel("Growth/year in %")
# Add legend
plt.legend(handles=[scatter_SS, SS_cen], loc='upper right')
plt.show()


# Load and transpose UK inflation data
uk_inflation_data = read_file('EuropeLand.csv')
uk_inf_data_trsp = uk_inflation_data.T

# Cleaning the transposed data
uk_inf_data_trsp.columns = ['Land']
uk_inf_data_trsp = uk_inf_data_trsp.drop('Year')
uk_inf_data_trsp.reset_index(inplace=True)
uk_inf_data_trsp.rename(columns={'index': 'Year'}, inplace=True)
uk_inf_data_trsp['Year'] = uk_inf_data_trsp['Year'].astype(int)
uk_inf_data_trsp['Land'] = uk_inf_data_trsp['Land'].astype(float)

# getting x and y values for modeling
x = uk_inf_data_trsp['Year'].values.astype(float)
y = uk_inf_data_trsp['Land'].values.astype(float)

# Fitting the polynomial model to the data
popt, pcov = curve_fit(polynomial, x, y)

# Calculate error ranges for original data
y_e = calculate_error(x, polynomial, popt, pcov)

# predict for future years and predict values
x_f = np.arange(max(x) + 1, 2031)
y_f = polynomial(x_f, *popt)

# Calculate error ranges for future predictions
y_f_e = calculate_error(x_f, polynomial, popt, pcov)

# Plotting the fitting data and predicted data
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='Historical Data')
plt.plot(x, polynomial(x, *popt), 'r-', 
         label='Fitted Polynomial Model')
plt.fill_between(x, polynomial(x, *popt) - 
    y_e, polynomial(x, *popt) + y_e, color='orange', 
    alpha=0.5, label='Confidence Interval for Historical Data')
plt.plot(x_f, y_f, 'g--', label='Future Predictions')
plt.fill_between(x_f, y_f - y_f_e, y_f +
                 y_f_e, color='lightgreen', 
                 alpha=0.5, label='Confidence Interval for Future Predictions')
plt.title('Fitting and forcasting for Europe Data')
plt.xlabel('Year')
plt.ylabel('Arable land[Hect per person]')
plt.legend()
plt.show()

