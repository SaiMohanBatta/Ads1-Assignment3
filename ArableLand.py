import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
from scipy.optimize import curve_fit
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

# Function to fetch data from a CSV file
def FetchData(CSV_Name):
    """
    Reads a CSV file and returns a DataFrame.

    Parameters:
    CSV_Name (str): The path to the CSV file to be read.

    Returns:
    DataFrame: A DataFrame containing the data read from the CSV file.
    """
    df = pd.read_csv(CSV_Name)
    return df


# Function to calculate silhouette score for clustering evaluation
def SilhouetteScore(xy, n_clusters):
    """
    Calculates the silhouette score for a given number of clusters.

    Parameters:
    xy (array): The data for clustering.
    n_clusters (int): The number of clusters to form.

    Returns:
    S_Score(float): The silhouette score
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    S_Score = silhouette_score(xy, labels)
    return S_Score


# Function for a polynomial fit
def poly_fit(x, a, b, c, d):
    """
    Function for calculating the value of a polynomial at x.

    Parameters:
    x (number or array): The value(s) at which the polynomial is evaluated.
    a, b, c, d (number): Coefficients of the polynomial.

    Returns:
    number or array: The value of the polynomial at x.
    """
    return a * x**3 + b * x**2 + c * x + d


# Function to calculate error range for a given function and parameters
def err_range(x, f, param, cov):
    """
    Calculates error for a given function and its parameters.

    Parameters:
    x (number or array): The input value(s) at which the error is evaluated.
    f (function): The function for which the error is calculated.
    param (list or array): Parameters of the function.
    cov (array): The covariance matrix of the parameters.

    Returns:
    numpy.ndarray: The calculated error values corresponding to each x.
    """
    var = np.zeros_like(x)
    for i in range(len(param)):
        deriv1 = deriv(x, f, param, i)
        for j in range(len(param)):
            deriv2 = deriv(x, f, param, j)
            var += deriv1 * deriv2 * cov[i, j]
    return np.sqrt(var)


# Function to calculate the derivative of a function
def deriv(x, f, param, idx):
    """
    Calculates the derivative of a function wrt to one of its parameters.

    Parameters:
    x (number or array): The value(s) at which the derivative is calculated.
    f (function): The function for which the derivative is calculated.
    param (list or array): Parameters of the function.
    idx (int): The index of the parameter

    Returns:
    numpy.ndarray or number: The deriv of the function 
    """
    val = 1e-6
    delta = np.zeros_like(param)
    delta[idx] = val * abs(param[idx])
    up_range = param + delta
    low_range = param - delta
    diff = 0.5 * (f(x, *up_range) - f(x, *low_range))
    return diff / (val * abs(param[idx]))


# Fetch the data and transform as required
df_land = FetchData("ArableLand.csv")
if df_land is not None:
    print(df_land.describe())

# Avoiding NaN values
df_land = df_land[(df_land["1960"].notna()) & (df_land["2020"].notna())]
df_land.reset_index(drop=True, inplace=True)

# Creating a DataFrame for the change in arable land
Incr = df_land[["Country Name", "1960"]].copy()
Incr["Change"] = 100.0 / 60.0 * (df_land["2020"] - 
                                 df_land["1960"]) / df_land["1960"]
warnings.filterwarnings("ignore", category=UserWarning)

# Displaying summary statistics for the change in arable land
print(Incr.describe())
print()
print(Incr.dtypes)

# Scatter plot for original data
plt.figure(figsize=(8, 8))
plt.scatter(Incr["1960"], Incr["Change"])
plt.xlabel("Arable land (hect per person), 1960")
plt.ylabel("Change per year in %")
plt.show()

# Normalization of data
scaler = RobustScaler()
df_clust = Incr[["1960", "Change"]]
scaler.fit(df_clust)
df_norm = scaler.transform(df_clust)

# Scatter plot for normalized data
plt.figure(figsize=(8, 8))
plt.scatter(df_norm[:, 0], df_norm[:, 1])
plt.xlabel("Normalized Arable land (hect per person), 1960")
plt.ylabel("Normalized Change per year [%]")
plt.show()

# Fetching silhouette score
for cluster_count in range(2, 11):
    Scr = SilhouetteScore(df_norm, cluster_count)
    S_message = f"The silhouette score for {cluster_count:3d} clusters is"
    print(f"{S_message} {Scr:7.4f}")

# Clustering with KMeans
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
kmeans.fit(df_norm)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
cen = scaler.inverse_transform(centers)

# Initial Scatter Plot with clusters
plt.figure(figsize=(8, 8))
sctr = plt.scatter(Incr["1960"], Incr["Change"], c=labels,
                   s=10, marker="o", cmap=cm.rainbow, label='Data Points')
cen_sctr = plt.scatter(cen[:, 0], cen[:, 1],
                       s=45, c="k", marker="d", label='Cluster Centers')
plt.xlabel("Arable land (hect per person), 1960")
plt.ylabel("Change per year in %")

# Adding legend
plt.legend(handles=[sctr, cen_sctr], loc='upper right')
plt.show()

print(cen)

# Fetching subset of original data
Incr_2 = Incr[labels == 0].copy()
print(Incr_2.describe())

# Normalizing the data within the subset
df_clust2 = Incr_2[["1960", "Change"]]
scaler.fit(df_clust2)
norm_dt2 = scaler.transform(df_clust2)

plt.figure(figsize=(8, 8))
plt.scatter(norm_dt2[:, 0], norm_dt2[:, 1])
plt.xlabel("Arable land (hect per person), 1960")
plt.ylabel("Change per year in %")
plt.show()

# Clustering within the subset
k_subset = cluster.KMeans(n_clusters=3, n_init=20)
k_subset.fit(norm_dt2)
Labels_SS = k_subset.labels_
subset_cen = k_subset.cluster_centers_

# Backscaling the cluster centers
cen_SS = scaler.inverse_transform(subset_cen)

# Second Scatter Plot (Within Subset):
plt.figure(figsize=(8.0, 8.0))
sctr_SS = plt.scatter(Incr_2["1960"], Incr_2["Change"],
                      c=Labels_SS, s=10, marker="o",
                      cmap=cm.rainbow,
                      label='Data Points')
Sctr_cen_SS = plt.scatter(cen_SS[:, 0],
                          cen_SS[:, 1],
                          s=45, c="k", marker="d",
                          label='Cluster Centers')
plt.xlabel("Arable land (hect per person), 1960")
plt.ylabel("Change per year in %")
# Add legend
plt.legend(handles=[sctr_SS, Sctr_cen_SS], loc='upper right')
plt.show()

# Code for Fitting Europe Arable land Data
# Load and transpose Europe land data
EU_Land_dt = FetchData('EuropeLand.csv')
EU_Land_dt_T = EU_Land_dt.T

# Cleaning the transposed data
EU_Land_dt_T.columns = ['Land']
EU_Land_dt_T = EU_Land_dt_T.drop('Year')
EU_Land_dt_T.reset_index(inplace=True)
EU_Land_dt_T.rename(columns={'index': 'Year'}, inplace=True)
EU_Land_dt_T['Year'] = EU_Land_dt_T['Year'].astype(int)
EU_Land_dt_T['Land'] = EU_Land_dt_T['Land'].astype(float)

# Appending to x and y values for modeling
x_val = EU_Land_dt_T['Year'].values.astype(float)
y_val = EU_Land_dt_T['Land'].values.astype(float)

# Fitting the polynomial model to the data
popt, pcov = curve_fit(poly_fit, x_val, y_val)

# Calculate error ranges for original data
y_err = err_range(x_val, poly_fit, popt, pcov)

# Predict for future years and predict values
fut_x = np.arange(max(x_val) + 1, 2041)
fut_y = poly_fit(fut_x, *popt)

# Calculate error ranges for predictions
y_fut_err = err_range(fut_x, poly_fit, popt, pcov)

# Plotting the fitting data and predicted data
plt.figure(figsize=(10, 6))
plt.plot(x_val, y_val, 'p-', label='Original Data')
plt.plot(x_val, poly_fit(x_val, *popt), 'g-',
         label='Fitted Model')
plt.fill_between(x_val, poly_fit(x_val, *popt) -
                 y_err, poly_fit(x_val, *popt) + y_err, color='lightgreen',
                 alpha=0.5, label='CI for Original Data')
plt.plot(fut_x, fut_y, 'r--', label='Predictions')
plt.fill_between(fut_x, fut_y - y_fut_err, fut_y +
                 y_fut_err, color='Orange',
                 alpha=0.5, label='CI for  Predictions')
plt.title('Fitting and forecasting for Europe Data')
plt.xlabel('Year')
plt.ylabel('Arable land [Hect per person]')
plt.legend()
plt.show()
