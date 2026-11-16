import numpy as np

np.random.seed(7)
readings = np.random.randint(20, 41, size=(6, 3))

print("Sensor readings:")
print(readings)

mean = readings.mean(axis=0)

print("Mean per sensor:", mean)
std = readings.std(axis=0)

standardized = (readings - mean) / std

print("Normalized column means:",
      np.round(standardized.mean(axis=0), 2))

print("Hour of max per sensor:",
      readings.argmax(axis=0))

new_readings = np.vstack([readings, mean])
print("New shape:", new_readings.shape)

np.random.seed(7)
readings_again = np.random.randint(20, 41, size=(6, 3))
print("Reproducible:", np.array_equal(readings, readings_again))
