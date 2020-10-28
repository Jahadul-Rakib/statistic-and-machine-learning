import numpy as np
import statistics as ss

data_set = [1, 2, 3, 4, 5, 6, 3, 3, 2, 42, 34, 2, 4, 6, 73]
# ---------------------------------------Measures of center

# mean (Average)
mean_data = ss.mean(data_set)
print("Mean: ", mean_data)

# median (Middle point)
median_data = ss.median(data_set)
print("Median: ", mean_data)

# Mod (Maximum exist number in dataset)
mode_data = ss.mode(data_set)
print("Mode: ", mode_data)

# --------------------------------------Measures of Spread

# Range (Basically difference between lowest value and highest value)
range_data = np.max(data_set) - np.min(data_set)
print("Range: ", range_data)

# Quartile
'''A quartile is a type of quartile which divides the number 
   of data points into four more or less equal parts, or quarters'''

quartile_lower_half = np.percentile(data_set, 25)  # Lower half quartile = first 25 % of data
print("Lower Quartile", quartile_lower_half)

quartile_middle = np.percentile(data_set, 50)  # median quartile = first 50 % of data
print("Median Quartile", quartile_middle)

quartile_upper_half = np.percentile(data_set, 75)  # upper half quartile = first 75 % of data
print("Upper Quartile", quartile_upper_half)

# Inter-quartile Range
interquartile_range = quartile_upper_half - quartile_lower_half
print("Interquartile Range", interquartile_range)

# Variance (it measures how far a set of numbers is spread out from their average value)
variance_data = np.var(data_set)
print("Variance", variance_data)

# Standard Deviation (The standard deviation is a summary measure of the differences of each observation from the mean)
standard_deviation = np.std(data_set);
print("Standard Deviation", standard_deviation)
