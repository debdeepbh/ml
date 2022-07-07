import pandas as pd

df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})

print(df)

out = pd.get_dummies(df, prefix=['col1', 'col2'])
print(out)
