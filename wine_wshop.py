import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('../data/wine.csv')

corr_mat = dataset.corr()
print('pearson correlation = \n', corr_mat, sep='')

plt.figure(figsize=(13,9))
ax = sns.heatmap(data=corr_mat, annot=True,
            cmap='GnBu', annot_kws={"weight": "bold"})
plt.title('Correlation Matrix for wine dataset')
plt.show()

target_column = corr_mat['Cultivar']
target_column.drop(['Cultivar'], inplace=True)

candidates = target_column[(target_column > 0.5) | (target_column < -0.5)]
print('candidates w.r.t. wine =\n', candidates, '\n', sep='')

to_drop = list(set(corr_mat.index) - set(candidates.index))
print('drop the following from corr_mat =', to_drop, '\n')

workset = corr_mat.drop(index=to_drop, columns=to_drop)
print('after dropping =\n', workset, '\n', sep='')

skip = []
accept = []
for colname in workset.columns:
	if not colname in skip and not colname in accept:
		series = workset[colname]

		# look for other features that are 
		# highly-correlated with the feature "colname"
		series = series[(series > 0.6)]

		# fetch the Pandas series from the 'candidates'
		# dataframe that only contains items found
		# in our 'series' variable (above)
		# alike = candidates.loc[series.index]
		alike = candidates[series.index]
		print('alike =\n', alike, '\n', sep='')

		# idxmax() to get the feature that is most 
		# correlated with "mpg" (our target). 
		# abs() to absolute the values because 
		# the features could be either positively
		# or negatively correlated to "mpg" (our target)
		top = alike.abs().idxmax()

		# accept the "top" feature
		accept += [top]
		
		# discard other highly-correlated features
		# with respect to the "top" feature 
		skip += set(alike.index) - set([top])

print('skip = ', skip)
print('selected =', accept)