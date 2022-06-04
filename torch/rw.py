import torch
import os
import pandas as pd
# source: https://d2l.ai/chapter_preliminaries/pandas.html


# Write a csv
# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# data_file = os.path.join('..', 'data', 'house_tiny.csv')
os.makedirs(os.path.join('data'), exist_ok=True)
data_file = os.path.join('data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# read the csv
data = pd.read_csv(data_file)
print(data)

# replace NaN with mean of the column
data['NumRooms'].fillna(value=data['NumRooms'].mean(), inplace=True)
print(data)

# data:
#    NumRooms Alley   Price
# 0       3.0  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       3.0   NaN  140000

# categorize Alley into Pave and NaN
# converts categorical variable into indicator variable
data = pd.get_dummies(data, dummy_na=True)
print(data)

#    NumRooms   Price  Alley_Pave  Alley_nan
# 0       3.0  127500           1          0
# 1       2.0  106000           0          1
# 2       4.0  178100           0          1
# 3       3.0  140000           0          1

## Convert subset of the table into tensors
X,y = torch.tensor(data.iloc[:,[0,2,3]].values), torch.tensor(data.iloc[:,1].values)
print('X=',X,'y=',y)
