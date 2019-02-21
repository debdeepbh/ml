# happiness vs GDP
import pandas as pd
bli =  pd.read_csv("datasets/BLI_20012019062939110.csv")

gdp = pd.read_csv("datasets/WEO_Data.xls", delimiter='\t', encoding='latin1',  na_values="n/a")

#print( bli[['LOCATION','Country','Value']])
print(list(gdp))
print(gdp['Country'])
print(bli['Country'])



