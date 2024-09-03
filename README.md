## DEV
### 3(i)
```
import numpy as np

a = np.array([1, 2, 3])
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
a[0] = 5
print(a)

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = a[:2, 1:3]
print(a[0, 1])
b[0, 0] = 77
print(a[0, 1])

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(a)

b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])

a[np.arange(4), b] += 10
print(a)

x = np.array([1, 2])
print(x.dtype)

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x + y)
print(np.add(x, y))

x = np.array([[1, 2], [3, 4]])
print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))
```
##
### 3(ii)
```
import pandas as pd

data = pd.DataFrame({
    "x1": ["y", "x", "y", "x", "x", "y"],
    "x2": range(16, 22),
    "x3": range(1, 7),
    "x4": ["a", "b", "c", "d", "e", "f"],
    "x5": range(30, 24, -1)
})

print(data)
```
##
```
data_row = data[data.x2 < 20]
print(data_row)

data_col = data.drop("x1", axis=1)
print(data_col)
```
##
```
data_col = data.drop("x1", axis=1)
print(data_col)
```
```
data_med = data["x5"].median()
print(data_med)
```
##
## 3.3
```
from matplotlib import pyplot as plt
import numpy as np

x = [20, 25, 37]  
y = [25000, 40000, 60000]  

plt.plot(x, y)
plt.xlabel("Age")
plt.ylabel('Salary')
plt.title('Salary by Age')
plt.show()
```
##
## 4 
```
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_data = load_iris(as_frame=True)
iris = iris_data.frame

# Display the first 5 rows of the dataset
print(iris.head())

# Summary statistics of the dataset
print(iris.describe())

# Calculate the mean of the Sepal Length
mean_sepal_length = iris['sepal length (cm)'].mean()
print("Mean of Sepal Length:", mean_sepal_length)

# Select only the 'sepal length (cm)' column
a = iris[['sepal length (cm)']]
print(a.head())

# Filter rows where 'sepal length (cm)' is greater than 5
c = iris[iris['sepal length (cm)'] > 5]
print(c)

# Plot Sepal Length using matplotlib
plt.plot(iris['sepal length (cm)'], 'o')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.title('Sepal Length Plot')
plt.show()
```
##
## 4
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the dataset using read_csv
df = pd.read_csv("stock_data.csv", parse_dates=["Date"], index_col="Date")

# Displaying the first five rows of the dataset
print(df.head())
```
```
from pandas import read_csv
from matplotlib import pyplot

# Read the dataset using read_csv
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

# Print the first five rows of the series
print(series.head())
```
```
from pandas import read_csv

from matplotlib import pyplot

series = read_csv('daily-minimum-temperatures.csv', header=O, index_col=O,parse_dates=True,

squeeze=True)

series.plot()

pyplot.show()
```
```
from pandas import read_csv
from matplotlib import pyplot

# Read the dataset using read_csv
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

# Plot the series with black dot markers
series.plot(style='k.')

# Show the plot
pyplot.show()
```
```
from pandas import read_csv

from matplotlib import pyplot

series = read_csv('daily-minimum-temperatures.csv', header=O, index_col=O, parse_dates=True, squeeze=True)

series.hist()

pyplot.show()
```
```
from pandas import read_csv

from matplotlib import pyplot

series = read_csv('daily-minimum-temperatures.csv', header=O, index_col=O, parse_dates=True, squeeze=True)

series.hist()

pyplot.show()
```
```
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot

# Read the dataset using read_csv
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

# Group by year
groups = series.groupby(Grouper(freq='A'))

# Create an empty DataFrame to store yearly data
years = DataFrame()

# Populate the DataFrame with yearly data
for name, group in groups:
    years[name.year] = group.values

# Plot a boxplot of the yearly data
years.boxplot()

# Show the plot
pyplot.show()
```
## 6
```
# Install necessary packages (this should be done in the command line, not in the script)
# pip install pyecharts
# pip install echarts-countries-pypkg
# pip install echarts-china-provinces-pypkg
# pip install echarts-china-cities-pypkg
# pip install echarts-china-counties-pypkg

import pyecharts
print(pyecharts.__version__)  # Use __version__ instead of version

import pandas as pd
from pyecharts.charts import Map
from pyecharts import options as opts

# Read data from the Excel file
data = pd.read_excel('GDP.xlsx')

# Prepare the data
province = list(data["province"])
gdp = list(data["2019_gdp"])  # Ensure the column name is correct and matches your Excel file

# Create a list of tuples for pyecharts
data_list = [list(z) for z in zip(province, gdp)]

# Create the map chart
c = (
    Map(init_opts=opts.InitOpts(width="1000px", height="600px"))  # Initialize map size
    .set_global_opts(
        title_opts=opts.TitleOpts(title="2019 Provinces GDP Distribution (unit: 100 million yuan)"),
        visualmap_opts=opts.VisualMapOpts(
            type_="continuous"  # Use "continuous" for gradient color mapping, "scatter" is not suitable here
        )
    )
    .add("GDP", data_list, maptype="china")  # Add data to the map
    .render("Map1.html")  # Render the map to an HTML file
)
```


