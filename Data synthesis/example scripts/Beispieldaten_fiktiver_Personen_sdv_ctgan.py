from sdv.tabular import CTGAN
import faker
import pandas as pd


faker.proxy.DEFAULT_LOCALE = 'de_AT'

df = pd.read_csv('Beispieldaten_fiktiver_Personen.csv', sep=';')
df['Vorname_Männlich'] = ''
df['Vorname_Weiblich'] = ''

model = CTGAN(anonymize_fields=
              {'Vorname_Männlich': 'first_name_male',
               'Vorname_Weiblich': 'first_name_female',
               'Nachname': 'last_name',
               'Straße': 'street_address'}
              )

model.fit(df)
synthetic_df = model.sample(num_rows=12)
synthetic_df['Vorname'] = ''

for index in synthetic_df.index:
    if synthetic_df['Geschlecht'][index] == 'Männlich':
        synthetic_df.at[index, 'Vorname'] = synthetic_df['Vorname_Männlich'][index]
    else:
        synthetic_df.at[index, 'Vorname'] = synthetic_df['Vorname_Weiblich'][index]

synthetic_df = synthetic_df.drop(columns=['Vorname_Männlich', 'Vorname_Weiblich'])
synthetic_df.to_csv('Beispieldaten_fiktiver_Personen_Output.csv', index=False, encoding='UTF-8', sep=';')

