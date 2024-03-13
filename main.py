import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# Task 1

list1 = [np.random.rand() for x in range(1000000)]
list2 = [np.random.rand() for y in range(1000000)]
array1 = np.array(list1)
array2 = np.array(list2)

# Multiplying two lists
start_time = time.perf_counter()
result_list = [a * b for a, b in zip(list1, list2)]
list_time = time.perf_counter() - start_time

# Multiplying two arrays
start_time = time.perf_counter()
result_array = np.multiply(array1, array2)
array_time = time.perf_counter() - start_time

print("Time taken for element wise multiplication using lists: ", list_time)
print("Time taken for element wise multiplication using NumPy arrays: ", array_time)

# Task 2 (Variant 8: data2.csv, Column: 4)

array = np.genfromtxt('data2.csv', delimiter=',')
array = array[1:]
Chloramines = np.array(array[:, 3], dtype=float)
Chloramines = Chloramines[~np.isnan(Chloramines)]

# Histogram
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot()
ax.hist(Chloramines, bins=50, color='lightblue', edgecolor='blue')
plt.grid()
plt.title('Histogram of Chloramines')
plt.xlabel('Chloramines')
plt.ylabel('Frequency')
plt.show()

# Normalized Histogram
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot()
ax.hist(Chloramines, bins=50, color='lightgreen', edgecolor='green', density=True)
plt.grid()
plt.title('Normalized Histogram of Chloramines')
plt.xlabel('Chloramines')
plt.ylabel('Frequency')
plt.show()

# Standard deviation
std_dev = np.std(Chloramines)
print(f'Standard Deviation of Chloramines values: {std_dev}')

# Task 3 (Variant 8: x∈(-3п;3п); y=cos(x); z=x/sin(x) )
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xa = np.linspace(-3 * np.pi, 3 * np.pi, 100)
ya = np.cos(xa)
za = xa / np.sin(xa)

ax.plot(xa, ya, za, c='blue')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()

# Additional Task

fig, ax = plt.subplots()

xax = np.linspace(0, 2 * np.pi, 100)
yax = np.sin(xax)

line, = ax.plot(xax, yax)


def update(frame):
    line.set_ydata(np.sin(xax + frame / 10))
    return line,


ani = FuncAnimation(fig, update, frames=100, blit=True, interval=50)
writer = PillowWriter(fps=50)
ani.save("sin_animation.gif", writer=writer)
plt.show()
