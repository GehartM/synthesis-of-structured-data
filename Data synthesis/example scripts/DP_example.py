from pydp.algorithms.laplacian import BoundedMean
from pydp.algorithms.laplacian import Count
from pydp.algorithms.laplacian import Max
from pydp.algorithms.laplacian import Min
import pandas as pd
import statistics

df = pd.read_csv('Beispieldaten_fiktiver_Personen.csv', sep=';')

# Information about the original data set
print('Anzahl an Einträgen: {0}'.format(df.count()['Alter']))
print('Durchschnittliches Alter: {0:.2f}'.format(statistics.mean(list(df['Alter']))))
print('Durchschnittliches Bruttogehalt: {0:.2f}'.format(statistics.mean(list(df['Bruttogehalt']))))
print('Minimales Alter: {0}'.format(df.min()['Alter']))
print('Maximales Alter: {0}'.format(df.max()['Alter']))

# Information about the DP version
dp_count = Count(1, dtype='int')
print('\nAnzahl an Einträgen DP: {0}'.format(dp_count.quick_result(list(df['Alter']))))
age_dp_mean = BoundedMean(1, lower_bound=0, upper_bound=100, dtype='float')
print('Durchschnittliches Alter DP: {0:.2f}'.format(age_dp_mean.quick_result(list(df['Alter']))))
income_dp_mean = BoundedMean(1, lower_bound=0, upper_bound=100000, dtype='float')
print('Durchschnittliches Bruttogehalt DP: {0:.2f}'.format(income_dp_mean.quick_result(list(df['Bruttogehalt']))))
age_dp_min = Min(1, lower_bound=0, upper_bound=100, dtype='int')
print('Minimales Alter DP: {0}'.format(age_dp_min.quick_result(list(df['Alter']))))
age_dp_max = Max(1, lower_bound=0, upper_bound=100, dtype='int')
print('Maximales Alter DP: {0}'.format(age_dp_max.quick_result(list(df['Alter']))))



