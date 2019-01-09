import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv('D:\\pers\\data\\bugs-1jan2018-31dec2018_firefox.csv')

data_frame.drop(['Bug ID', 'Product', 'Assignee', 'Status', 'Resolution', 'Summary', 'Changed', 'Opened'], axis = 1, inplace = True)

series = data_frame['Component'].value_counts()
print(series)
series.plot.bar()
plt.title('Components 2018')
plt.show()

