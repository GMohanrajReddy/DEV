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
