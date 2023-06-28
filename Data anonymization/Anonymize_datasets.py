import pandas as pd
import os
import sys
import csv
from pathlib import Path
import datetime
import numpy as np
import re
from pycanon import anonymity, report
from sklearn import tree, svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from dateutil.relativedelta import relativedelta
import datetime
from sklearn.impute import SimpleImputer
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.tree import export_graphviz
import calendar


input_folder = Path('Input')
output_folder = Path('Output')
log_file_handler = open('Anonymize_datasets_{}.log'.format(str(datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))), 'w')


class UnbufferedLogging:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        log_file_handler.write(data)

    def flush(self):
        log_file_handler.flush()


def check_if_required_dirs_exist():
    check_if_dir_exists(output_folder)
    if not check_if_dir_exists(input_folder):
        print('\n\t[!] Provide the input csv-files inside the newly created "{}"-folder.\n\t[!] Afterwards start the script again.'.format(input_folder))
        sys.exit(1)


def check_if_dir_exists(folder_name):
    if not os.path.exists(folder_name):
        print('\t[-] "{}"-folder was created.'.format(folder_name))
        os.makedirs(folder_name)
        return False
    return True


def check_if_file_exists(file_name):
    if not os.path.exists(file_name):
        return False
    return True


def get_list_of_input_csv():
    print('\n\t[-] Scanning "{}"-folder for csv-files...'.format(input_folder))
    files_in_dir = os.listdir(input_folder)
    csv_files = list(filter(lambda f: f.endswith('.csv'), files_in_dir))
    if len(csv_files) > 0:
        for csv_file in csv_files:
            print('\t\t[.] Found "{}"'.format(csv_file))
    else:
        print('\t[!] No csv-files were found inside the folder "{}"!'.format(input_folder))
        sys.exit(1)
    return csv_files


def get_delimiter_of_csv(input_file, bytes_to_read=9000):
    sniffer = csv.Sniffer()
    data = open(input_file, 'r').read(bytes_to_read)
    try:
        delimiter = sniffer.sniff(data).delimiter
    except:
        delimiter = None
    return delimiter


def read_csv_file(input_file):
    input_file = input_folder / input_file
    delimiter = get_delimiter_of_csv(input_file)
    if delimiter is None:
        print('\n\t[!] The delimiter for "{}" could not be automatically determined!'.format(input_file))
        while delimiter is None:
            delimiter = input('\tEnter the delimiter for this file: ')
            if len(delimiter) != 1:
                print('\t[!] The delimiter can only be one character!')
                delimiter = None
    try:
        return pd.read_csv(input_file, sep=delimiter, decimal=',')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(input_file, sep=delimiter, decimal=',', encoding='utf-8-sig')
        except UnicodeDecodeError:
            return pd.read_csv(input_file, sep=delimiter, decimal=',', encoding='ISO-8859-1')


def get_random_number_list(unique_value_list):
    return np.random.choice(range(1, len(unique_value_list)*5), size=len(unique_value_list), replace=False).tolist()


def anonymize_data(input_file):
    print('\n\t[-] Starting the anonymization process for "{}"...'.format(input_file))
    df = read_csv_file(input_file)

    if df is not None:
        if input_file == 'Faker_Jö_Bonusclub_Output.csv':
            df = anonymize_joe_bonusclub(df)
        elif input_file == 'Faker_Elektronischer_Impfpass_Output.csv':
            df = anonymize_vaccination_record(df)
        elif input_file == 'Online Retail.csv':
            df = anonymize_online_retail(df)
        elif input_file == 'Case_Information.csv':
            df = anonymize_case_information(df)
        else:
            print('\t\t[!] No anonymization routine for "{}" is implemented!\n\t\t[!] Skipping "{}".'.format(input_file, input_file))
            return

    output_file = save_anonymous_data_as_csv(df, input_file)
    print('\t\t[.] Saved anonymous dataset to "{}".'.format(output_file))
    return df


def get_k_anonymity_l_diversity_t_closeness(df, quasi_identifier, sensitive_attributes, only_k_anonymity=False):
    print('\t\t[.] K-Anonymity: {}'.format(anonymity.k_anonymity(df, quasi_identifier)))
    if not only_k_anonymity:
        print('\t\t[.] L-Diversity: {}'.format(anonymity.l_diversity(df, quasi_identifier, sensitive_attributes)))
        print('\t\t[.] T-Closeness: {}'.format(anonymity.t_closeness(df, quasi_identifier, sensitive_attributes)))


def map_seasons_to_date(month):
    months_per_season = [['12', '01', '02'], ['03', '04', '05'],
             ['06', '07', '08'], ['09', '10', '11']]

    if month in months_per_season[0]:
        return 'Winter'
    elif month in months_per_season[1]:
        return 'Frühling'
    elif month in months_per_season[2]:
        return 'Sommer'
    elif month in months_per_season[3]:
        return 'Herbst'
    else:
        return np.nan


def anonymize_joe_bonusclub(df):
    df.drop('Vorname', inplace=True, axis=1)
    df.drop('Nachname', inplace=True, axis=1)
    df.drop('Straße', inplace=True, axis=1)
    df.drop('Telefonnummer', inplace=True, axis=1)
    df.drop('E-Mailadresse', inplace=True, axis=1)
    df.drop('Postleitzahl', inplace=True, axis=1)
    df['Geburtsdatum'] = pd.to_datetime(df['Geburtsdatum'], format='%Y-%m-%d')
    df['Geburtsdatum'] = pd.DatetimeIndex(df['Geburtsdatum']).year
    df['Geburtsdatum'] = 2023 - df['Geburtsdatum']
    df['Geburtsdatum_Kategorie'] = pd.cut(x=df['Geburtsdatum'], bins=[-1, 14, 29, 44, 59, 74, 101], labels=['<15', '15-29', '30-44', '45-59', '60-74', '>74'])
    df.drop('Geburtsdatum', inplace=True, axis=1)
    df['Kaufdatum'] = pd.to_datetime(df['Kaufdatum'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')
    df = df.groupby(['Geschlecht', 'Geburtsdatum_Kategorie', 'Kaufdatum']).filter(lambda x: len(x) > 400).reset_index()

    unique_values = df['Mitgliedsnummer'].unique()
    mapping = get_random_number_list(df['Mitgliedsnummer'].unique())
    df['Mitgliedsnummer'] = df['Mitgliedsnummer'].replace(unique_values, mapping)
    df_no_duplicates = df.drop_duplicates(subset='Mitgliedsnummer', keep='first')

    quasi_identifier = ['Geschlecht', 'Geburtsdatum_Kategorie', 'Kaufdatum']
    sensitive_attributes = ['Gesamtpreis']
    get_k_anonymity_l_diversity_t_closeness(df_no_duplicates, quasi_identifier, sensitive_attributes, only_k_anonymity=True)
    return df


def anonymize_vaccination_record(df):
    df.drop('Vorname', inplace=True, axis=1)
    df.drop('Nachname', inplace=True, axis=1)
    df.drop('Sozialversicherungsnummer', inplace=True, axis=1)
    df.drop('Ort', inplace=True, axis=1)
    df.drop('Straße', inplace=True, axis=1)
    df['Geburtsdatum'] = pd.to_datetime(df['Geburtsdatum'], format='%Y-%m-%d')
    df['Geburtsdatum'] = pd.DatetimeIndex(df['Geburtsdatum']).year
    df['Geburtsdatum'] = 2023 - df['Geburtsdatum']
    df['Geburtsdatum_Kategorie'] = pd.cut(x=df['Geburtsdatum'], bins=[-1, 14, 29, 44, 59, 74, 101], labels=['<15', '15-29', '30-44', '45-59', '60-74', '>74'])
    age_category_column = df.pop("Geburtsdatum_Kategorie")
    df.insert(1, "Geburtsdatum_Kategorie", age_category_column)
    df.drop('Geburtsdatum', inplace=True, axis=1)

    df['Tetanus und Diphterie Impfdatum'] = pd.to_datetime(df['Tetanus und Diphterie Impfdatum'], format='%Y-%m-%d').dt.strftime('%m')
    df['Polio Impfdatum'] = pd.to_datetime(df['Polio Impfdatum'], format='%Y-%m-%d').dt.strftime('%m')
    df['FSME Impfdatum'] = pd.to_datetime(df['FSME Impfdatum'], format='%Y-%m-%d').dt.strftime('%m')
    df['Covid Impfdatum'] = pd.to_datetime(df['Covid Impfdatum'], format='%Y-%m-%d').dt.strftime('%m')
    df['Tetanus und Diphterie Impfdatum'] = df['Tetanus und Diphterie Impfdatum'].map(lambda x: map_seasons_to_date(x))
    df['Polio Impfdatum'] = df['Polio Impfdatum'].map(lambda x: map_seasons_to_date(x))
    df['FSME Impfdatum'] = df['FSME Impfdatum'].map(lambda x: map_seasons_to_date(x))
    df['Covid Impfdatum'] = df['Covid Impfdatum'].map(lambda x: map_seasons_to_date(x))

    df.drop('Tetanus und Diphterie Impfdatum', inplace=True, axis=1)
    df.drop('Polio Impfdatum', inplace=True, axis=1)
    df.drop('FSME Impfdatum', inplace=True, axis=1)
    df.drop('Covid Impfdatum', inplace=True, axis=1)
    """
    df['Tetanus und Diphterie Impfdatum'] = pd.to_datetime(df['Tetanus und Diphterie Impfdatum'], format='%Y-%m-%d').dt.strftime('%B %Y')
    df['Polio Impfdatum'] = pd.to_datetime(df['Polio Impfdatum'], format='%Y-%m-%d').dt.strftime('%B %Y')
    df['FSME Impfdatum'] = pd.to_datetime(df['FSME Impfdatum'], format='%Y-%m-%d').dt.strftime('%B %Y')
    df['Covid Impfdatum'] = pd.to_datetime(df['Covid Impfdatum'], format='%Y-%m-%d').dt.strftime('%B %Y')
    """

    unique_values = df['Postleitzahl'].unique()
    mapping = list()
    for value in unique_values:
        value = str(value)
        value = re.sub(r'7[0-9]{3}', value[0], value)
        value = re.sub(r'9[0-8][0-9]{2}', value[:2], value)
        value = re.sub(r'[2,3][0-9]{3}', value[0], value)
        value = re.sub(r'4[0-9]{3}', value[0], value)
        value = re.sub(r'5[0-9]{3}', value[0], value)
        value = re.sub(r'8[0-9]{3}', value[0], value)
        value = re.sub(r'6[0-6][0-9]{2}', value[:2], value)
        value = re.sub(r'99[0-9]{2}', value[:2], value)
        value = re.sub(r'6[7-9][0-9]{2}', value[:2], value)
        value = re.sub(r'1[0-9]{3}', value[0], value)
        mapping.append(value)
    df['Postleitzahl'] = df['Postleitzahl'].replace(unique_values, mapping)
    df.drop('Postleitzahl', inplace=True, axis=1)

    df['Tetanus und Diphterie Impfstoffhersteller'] = df['Tetanus und Diphterie Impfstoffhersteller'].astype(str)
    df['Polio Impfstoffhersteller'] = df['Polio Impfstoffhersteller'].astype(str)
    df['FSME Impfstoffhersteller'] = df['FSME Impfstoffhersteller'].astype(str)
    df['Covid Impfstoffhersteller'] = df['Covid Impfstoffhersteller'].astype(str)
    df['Tetanus und Diphterie Impfstoff'] = df['Tetanus und Diphterie Impfstoff'].astype(str)
    df['Polio Impfstoff'] = df['Polio Impfstoff'].astype(str)
    df['FSME Impfstoff'] = df['FSME Impfstoff'].astype(str)
    df['Covid Impfstoff'] = df['Covid Impfstoff'].astype(str)

    df = df.groupby(['Geschlecht', 'Geburtsdatum_Kategorie', 'Bundesland']).filter(lambda x: len(x) > 15).reset_index()

    quasi_identifier = ['Geschlecht', 'Geburtsdatum_Kategorie', 'Bundesland']
    sensitive_attributes = ['Tetanus und Diphterie Impfung', 'Polio Impfung', 'FSME Impfung', 'Covid Impfung', 'Tetanus und Diphterie Impfstoffhersteller', 'Polio Impfstoffhersteller', 'FSME Impfstoffhersteller', 'Covid Impfstoffhersteller', 'Tetanus und Diphterie Impfstoff', 'Polio Impfstoff', 'FSME Impfstoff', 'Covid Impfstoff']
    get_k_anonymity_l_diversity_t_closeness(df, quasi_identifier, sensitive_attributes)
    return df


def anonymize_online_retail(df):
    unique_values = df['CustomerID'].unique()
    mapping = get_random_number_list(df['CustomerID'].unique())
    df['CustomerID'] = df['CustomerID'].replace(unique_values, mapping)
    df = df.groupby('Country').filter(lambda x: len(x) > 5)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d.%m.%Y %H:%M').dt.strftime('%B %Y')
    df = df.groupby(['Country', 'InvoiceDate']).filter(lambda x: len(x) > 15).reset_index()

    df_no_duplicates = df.drop_duplicates('CustomerID', keep='first')
    no_duplicates_by_country = df_no_duplicates.groupby('Country').size()

    # Remove also all country were there are less than eleven deduplicated Customer IDs
    for country, occurence in no_duplicates_by_country.items():
        if occurence < 11:
            df = df[df['Country'] != country]
            df_no_duplicates = df_no_duplicates[df_no_duplicates['Country'] != country]

    quasi_identifier = ['Country', 'InvoiceDate']
    sensitive_attributes = list()
    get_k_anonymity_l_diversity_t_closeness(df_no_duplicates, quasi_identifier, sensitive_attributes, only_k_anonymity=True)
    return df


def anonymize_case_information(df):
    df['age_category'] = pd.cut(x=df['age'], bins=[-1, 14, 29, 44, 59, 74, 101], labels=['<15', '15-29', '30-44', '45-59', '60-74', '>74'])
    age_category_column = df.pop("age_category")
    df.insert(1, "age_category", age_category_column)
    df.drop('age', inplace=True, axis=1)
    df['date_announced'] = pd.to_datetime(df['date_announced'], format='%d.%m.%Y').dt.strftime('%B %Y')
    df['date_recovered'] = pd.to_datetime(df['date_recovered'], format='%d.%m.%Y').dt.strftime('%B %Y')
    df['date_of_death'] = pd.to_datetime(df['date_of_death'], format='%d.%m.%Y').dt.strftime('%B %Y')
    df['date_announced_as_removed'] = pd.to_datetime(df['date_announced_as_removed'], format='%d.%m.%Y').dt.strftime('%B %Y')
    df['date_of_onset_of_symptoms'] = pd.to_datetime(df['date_of_onset_of_symptoms'], format='%d.%m.%Y').dt.strftime('%B %Y')

    df = df.loc[~((df['age_category'] == '<15') & (df['sex'] == 'Female') & (df['date_announced'] == 'May 2020') & (df['status'] == 'For validation') & (df['region'] == 'Central Visayas (Region VII)'))]
    df = df.loc[~((df['age_category'] == '15-29') & (df['sex'] == 'Male') & (df['date_announced'] == 'May 2020') & (df['status'] == 'For validation') & (df['region'] == 'CALABARZON (Region IV-A)'))]
    df = df.loc[~((df['age_category'] == '15-29') & (df['date_announced'] == 'May 2020') & (df['status'] == 'For validation') & (df['region'] == 'Central Visayas (Region VII)'))]
    df = df.loc[~((df['age_category'] == '45-59') & (df['sex'] == 'Female') & (df['date_announced'] == 'May 2020') & (df['status'] == 'For validation') & (df['region'] == 'Central Visayas (Region VII)'))]
    df = df.loc[~((df['age_category'] == '60-74') & (df['date_announced'] == 'May 2020') & (df['status'] == 'For validation') & (df['region'] == 'Central Visayas (Region VII)'))]

    df.drop('date_recovered', inplace=True, axis=1)
    df.drop('date_of_death', inplace=True, axis=1)
    df.drop('date_announced_as_removed', inplace=True, axis=1)
    df.drop('province', inplace=True, axis=1)
    df.drop('muni_city', inplace=True, axis=1)
    df.drop('health_status', inplace=True, axis=1)
    df.drop('home_quarantined', inplace=True, axis=1)
    df.drop('date_of_onset_of_symptoms', inplace=True, axis=1)
    df.drop('pregnant', inplace=True, axis=1)
    df = df.groupby(['age_category', 'sex', 'date_announced', 'region']).filter(lambda x: len(x) > 15).reset_index()

    df['status'] = df['status'].astype(str)
    df['age_category'] = df['age_category'].astype(str)
    df['sex'] = df['sex'].astype(str)
    df['date_announced'] = df['date_announced'].astype(str)
    df['region'] = df['region'].astype(str)

    quasi_identifier = ['age_category', 'sex', 'date_announced', 'region']
    sensitive_attributes = ['status']
    get_k_anonymity_l_diversity_t_closeness(df, quasi_identifier, sensitive_attributes)
    return df


def save_anonymous_data_as_csv(df, output_filename):
    output_filename = output_folder / '{}'.format(output_filename)
    df.to_csv(output_filename, index=False, encoding='utf-8-sig', sep=';', decimal=',')
    return output_filename


def get_column_types(csv_file_name):
    if csv_file_name == 'Case_Information.csv':
        datetime_columns = ['date_announced', 'date_recovered', 'date_of_death', 'date_announced_as_removed', 'date_of_onset_of_symptoms']
        continuous_columns = ['age']
        categorical_columns = ['sex', 'status', 'province', 'muni_city', 'health_status', 'region', 'home_quarantined', 'pregnant']
    elif csv_file_name == 'Faker_Jö_Bonusclub_Output.csv':
        datetime_columns = ['Kaufdatum', 'Geburtsdatum']
        continuous_columns = ['Preis in € je Stk', 'Anzahl', 'Gesamtpreis', 'Gesammelte Ös']
        categorical_columns = ['Produktname', 'Kategorie', 'Verbrauchausgaben-Kategorie', 'Bezahlart', 'Geschlecht', 'Postleitzahl', 'Ort']
    elif csv_file_name == 'Online Retail.csv':
        datetime_columns = ['InvoiceDate']
        continuous_columns = ['Quantity', 'UnitPrice', 'CustomerID']
        categorical_columns = ['StockCode', 'Description', 'Country']
    elif csv_file_name == 'Faker_Elektronischer_Impfpass_Output.csv':
        datetime_columns = ['Geburtsdatum', 'Tetanus und Diphterie Impfdatum', 'Tetanus und Diphterie Ablaufdatum des Impfstoffs', 'Polio Impfdatum', 'Polio Ablaufdatum des Impfstoffs', 'FSME Impfdatum', 'FSME Ablaufdatum des Impfstoffs', 'Covid Impfdatum', 'Covid Ablaufdatum des Impfstoffs']
        continuous_columns = []
        categorical_columns = ['Geschlecht', 'Bundesland', 'Postleitzahl', 'Ort', 'Tetanus und Diphterie Impfstoff', 'Tetanus und Diphterie Impfstoffhersteller', 'Polio Impfstoff', 'Polio Impfstoffhersteller', 'FSME Impfstoff', 'FSME Impfstoffhersteller', 'Covid Impfstoff', 'Covid Impfstoffhersteller']
    else:
        print('\n\t[!] Got unsupported csv-file "{}"!'.format(csv_file_name))
        sys.exit(1)
    return datetime_columns, continuous_columns, categorical_columns


def write_df_to_excel(results_df, file_name, section_name, execution_counter=1):
    section_name = section_name[:30]
    output_file = output_folder / file_name
    if execution_counter != 0 and check_if_file_exists(output_file):
        mode = 'a'
    else:
        mode = 'w'

    with pd.ExcelWriter(output_file, engine='openpyxl', mode=mode) as writer:
        results_df.to_excel(writer, sheet_name=section_name)
    print('\n\t\t[-] Results saved to "{}".'.format(output_file))


def check_applicability_for_csv(anonymous_df, input_file, counter):
    print('\n\t[-] Checking applicability for "{}"...'.format(input_file))
    original_df = read_csv_file(input_file)
    original_df = convert_specific_columns(original_df, input_file)
    original_df = add_modified_date_column(original_df, input_file)

    if original_df is not None and anonymous_df is not None:
        datetime_columns, continuous_columns, categorical_columns = get_column_types(input_file)
        applicability_results_df = check_applicability(original_df, anonymous_df, input_file, datetime_columns, continuous_columns, categorical_columns)
        write_df_to_excel(applicability_results_df, 'Validation_result_applicability.xlsx', input_file[:-4], counter)


def check_applicability(original_df, anonymous_df, csv_file_name, datetime_columns, continuous_columns, categorical_columns):
    csv_file_name = str(csv_file_name)
    df = pd.DataFrame()

    df['CSV-Datei'] = [csv_file_name]
    df.set_index('CSV-Datei', inplace=True)

    if re.search('.*{}'.format('Case_Information.csv'), csv_file_name):
        # Calculate accuracy and f1 score using Decision Tree, Support Vector Machine, Logistic Regression and Random_Forest
        # 1.0 good prediction, 0.0 bad prediction

        feature_columns_original = ['age', 'sex', 'pregnant', 'region']
        feature_columns_anonym = ['age_category', 'sex', 'region']
        target_column = 'status'
        categorical_columns.append('age_category')
        original_df = original_df[original_df[target_column].isin(['Recovered', 'Died']) == True]
        anonymous_df = anonymous_df[anonymous_df[target_column].isin(['Recovered', 'Died']) == True]

        df = calculate_accuracy_and_f1_score_for_applicability(df, 'Accuracy_Original', 'F1-Score_Original', 'Accuracy_Anonym', 'F1-Score_Anonym', original_df, anonymous_df, categorical_columns, feature_columns_original, feature_columns_anonym, target_column, continuous_columns)
    elif re.search('.*{}'.format('Faker_Jö_Bonusclub_Output.csv'), csv_file_name):
        # Identify association rules of bought items using the Apriori algorithm
        # 1.0 all association rules could be preserved, 0.0 no association rules were preserved

        output_file_name = 'Validation_result_applicability_basket_analysis_{}.xlsx'.format(csv_file_name[:-4])
        assoc_rules_orig = basket_analysis(original_df, csv_file_name, 'Mitgliedsnummer', 'Produktname', 'Anzahl', output_file_name, 'original', 0)
        assoc_rules_anonym = basket_analysis(anonymous_df, csv_file_name, 'Mitgliedsnummer', 'Produktname', 'Anzahl', output_file_name, 'anonym')
        df['Matched_association_rule_percentage'] = get_percentage_of_similar_association_rules(assoc_rules_orig, assoc_rules_anonym)
    elif re.search('.*{}'.format('Online Retail.csv'), csv_file_name):
        # Identify association rules of bought items using the Apriori algorithm
        # 1.0 all association rules could be preserved, 0.0 no association rules were preserved

        output_file_name = 'Validation_result_applicability_basket_analysis_{}.xlsx'.format(csv_file_name[:-4])
        assoc_rules_orig = basket_analysis(original_df, csv_file_name, 'InvoiceNo', 'Description', 'Quantity', output_file_name, 'original', 0)
        assoc_rules_anonym = basket_analysis(anonymous_df, csv_file_name, 'InvoiceNo', 'Description', 'Quantity', output_file_name, 'anonym')
        df['Matched_association_rule_percentage'] = get_percentage_of_similar_association_rules(assoc_rules_orig, assoc_rules_anonym)
    elif re.search('.*{}'.format('Faker_Elektronischer_Impfpass_Output.csv'), csv_file_name):
        # Calculate accuracy and f1 score using Decision Tree, Support Vector Machine, Logistic Regression and Random_Forest
        # 1.0 good prediction, 0.0 bad prediction

        feature_columns_original = ['Geschlecht', 'Geburtsdatum_year']
        feature_columns_anonym = ['Geschlecht', 'Geburtsdatum_Kategorie']
        target_column = 'Tetanus und Diphterie Impfung'
        continuous_columns.append('Geburtsdatum_year')
        categorical_columns.append('Geburtsdatum_Kategorie')

        df = calculate_accuracy_and_f1_score_for_applicability(df, 'Accuracy_Original_Tet_Diph', 'F1-Score_Original_Tet_Diph', 'Accuracy_Anonym_Tet_Diph', 'F1-Score_Anonym_Tet_Diph', original_df, anonymous_df, categorical_columns, feature_columns_original, feature_columns_anonym, target_column, continuous_columns)

        target_column = 'Polio Impfung'
        df = calculate_accuracy_and_f1_score_for_applicability(df, 'Accuracy_Original_Polio', 'F1-Score_Original_Polio', 'Accuracy_Anonym_Polio', 'F1-Score_Anonym_Polio', original_df, anonymous_df, categorical_columns, feature_columns_original, feature_columns_anonym, target_column, continuous_columns)

        target_column = 'FSME Impfung'
        df = calculate_accuracy_and_f1_score_for_applicability(df, 'Accuracy_Original_FSME', 'F1-Score_Original_FSME', 'Accuracy_Anonym_FSME', 'F1-Score_Anonym_FSME', original_df, anonymous_df, categorical_columns, feature_columns_original, feature_columns_anonym, target_column, continuous_columns)

        target_column = 'Covid Impfung'
        df = calculate_accuracy_and_f1_score_for_applicability(df, 'Accuracy_Original_Cov', 'F1-Score_Original_Cov', 'Accuracy_Anonym_Cov', 'F1-Score_Anonym_Cov', original_df, anonymous_df, categorical_columns, feature_columns_original, feature_columns_anonym, target_column, continuous_columns)
    else:
        print('\n\t[!] Got unsupported csv-file "{}"!'.format(csv_file_name))
        sys.exit(1)
    return df


def encode_data_with_zero_and_one(value):
    if value >= 1:
        return 1
    else:
        return 0


def basket_analysis(df, input_file, invoice_number_column, item_name_column, item_quantity_column, output_file_name, type, execution_counter=1):
    min_support = 0.03
    if input_file == 'Faker_Jö_Bonusclub_Output.csv':
        min_support = 0.01

    df.drop(df[df[item_quantity_column] <= 0].index, inplace=True)
    bought_items = df.groupby([invoice_number_column, item_name_column])[item_quantity_column].sum().unstack().reset_index().fillna(0).set_index(invoice_number_column)
    bought_items = bought_items.applymap(encode_data_with_zero_and_one)
    bought_items = bought_items[(bought_items > 0).sum(axis=1) > 1]
    frequent_itemsets = apriori(bought_items, min_support=min_support, use_colnames=True).sort_values("support", ascending=False)
    assoc_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1).sort_values("lift", ascending=False).reset_index(drop=True)

    write_df_to_excel(assoc_rules, output_file_name, type, execution_counter=execution_counter)
    return assoc_rules


def get_percentage_of_similar_association_rules(assoc_rules_orig, assoc_rules_anonym):
    antecedents_anonym = list(assoc_rules_anonym['antecedents'])
    matched_rules_count = 0

    if len(assoc_rules_orig) > 0:
        for index, row in assoc_rules_orig.iterrows():
            match = 0
            if row['antecedents'] in antecedents_anonym:
                matched_rows_anonym = assoc_rules_anonym.loc[assoc_rules_anonym['antecedents'] == row['antecedents']]
                for index_anoynm, row_anonym in matched_rows_anonym.iterrows():
                    if row_anonym['consequents'] == row['consequents']:
                        match = 1
                matched_rules_count += match

        matched_rules_percentage = matched_rules_count / len(assoc_rules_orig)
        return matched_rules_percentage
    else:
        return 0


def calculate_accuracy_and_f1_score_for_applicability(result_df, accuracy_orig_col_name, f1_score_orig_col_name, accuracy_anonym_col_name, f1_score_anonym_col_name, original_df, anonymous_df, categorical_columns, feature_columns_original, feature_columns_anonym, target_column, continuous_columns):
    all_relevant_columns_original = feature_columns_original.copy()
    all_relevant_columns_anonym = feature_columns_anonym.copy()
    all_relevant_columns_original.append(target_column)
    all_relevant_columns_anonym.append(target_column)
    all_relevant_categorical_columns_original = list(set(categorical_columns) & set(all_relevant_columns_original))
    all_relevant_categorical_columns_anonym = list(set(categorical_columns) & set(all_relevant_columns_anonym))
    all_relevant_continuous_columns_original = list(set(continuous_columns) & set(all_relevant_columns_original))
    all_relevant_continuous_columns_anonym = list(set(continuous_columns) & set(all_relevant_columns_anonym))

    original_df = original_df[all_relevant_columns_original].copy()
    anonymous_df = anonymous_df[all_relevant_columns_anonym].copy()
    mapping_original, inverted_mapping_original = get_mapping_for_categorical_fields_to_numerical(original_df, all_relevant_categorical_columns_original)
    mapping_anonym, inverted_mapping_anonym = get_mapping_for_categorical_fields_to_numerical(anonymous_df, all_relevant_categorical_columns_anonym)
    map_categorical_fields_to_numerical_values(original_df, all_relevant_categorical_columns_original, mapping_original)
    map_categorical_fields_to_numerical_values(anonymous_df, all_relevant_categorical_columns_anonym, mapping_anonym)
    original_df = original_df.dropna()
    anonymous_df = anonymous_df.dropna()

    accuracy_orig_result, f1_score_orig_result = generate_decision_tree_and_evaluate(original_df, feature_columns_original, target_column)
    accuracy_anonym_result, f1_score_anoynm_result = generate_decision_tree_and_evaluate(anonymous_df, feature_columns_anonym, target_column)
    result_df['Decision_Tree_{}'.format(accuracy_orig_col_name)] = [accuracy_orig_result]
    result_df['Decision_Tree_{}'.format(f1_score_orig_col_name)] = [f1_score_orig_result]
    result_df['Decision_Tree_{}'.format(accuracy_anonym_col_name)] = [accuracy_anonym_result]
    result_df['Decision_Tree_{}'.format(f1_score_anonym_col_name)] = [f1_score_anoynm_result]

    accuracy_orig_result, f1_score_orig_result = generate_support_vector_machine_and_evaluate(original_df, feature_columns_original, target_column)
    accuracy_anonym_result, f1_score_anoynm_result = generate_support_vector_machine_and_evaluate(anonymous_df, feature_columns_anonym, target_column)
    result_df['SVM_{}'.format(accuracy_orig_col_name)] = [accuracy_orig_result]
    result_df['SVM_{}'.format(f1_score_orig_col_name)] = [f1_score_orig_result]
    result_df['SVM_{}'.format(accuracy_anonym_col_name)] = [accuracy_anonym_result]
    result_df['SVM_{}'.format(f1_score_anonym_col_name)] = [f1_score_anoynm_result]

    accuracy_orig_result, f1_score_orig_result = generate_logistic_regression_and_evaluate(original_df, feature_columns_original, target_column)
    accuracy_anonym_result, f1_score_anoynm_result = generate_logistic_regression_and_evaluate(anonymous_df, feature_columns_anonym, target_column)
    result_df['Log_Regression_{}'.format(accuracy_orig_col_name)] = [accuracy_orig_result]
    result_df['Log_Regression_{}'.format(f1_score_orig_col_name)] = [f1_score_orig_result]
    result_df['Log_Regression_{}'.format(accuracy_anonym_col_name)] = [accuracy_anonym_result]
    result_df['Log_Regression_{}'.format(f1_score_anonym_col_name)] = [f1_score_anoynm_result]

    accuracy_orig_result, f1_score_orig_result = generate_random_forest_and_evaluate(original_df, feature_columns_original, target_column)
    accuracy_anonym_result, f1_score_anoynm_result = generate_random_forest_and_evaluate(anonymous_df, feature_columns_anonym, target_column)
    result_df['Random_Forest_{}'.format(accuracy_orig_col_name)] = [accuracy_orig_result]
    result_df['Random_Forest_{}'.format(f1_score_orig_col_name)] = [f1_score_orig_result]
    result_df['Random_Forest_{}'.format(accuracy_anonym_col_name)] = [accuracy_anonym_result]
    result_df['Random_Forest_{}'.format(f1_score_anonym_col_name)] = [f1_score_anoynm_result]
    return result_df


def generate_decision_tree_and_evaluate(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]

    if len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        dtree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        dtree = dtree.fit(X_train, y_train)
        y_prediction = dtree.predict(X_test)

        accuracy_result = metrics.accuracy_score(y_test, y_prediction)
        f1_score_result = metrics.f1_score(y_test, y_prediction, average='weighted')
        return accuracy_result, f1_score_result
    else:
        return np.nan, np.nan


def generate_support_vector_machine_and_evaluate(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]

    if len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_prediction = clf.predict(X_test)

        accuracy_result = metrics.accuracy_score(y_test, y_prediction)
        f1_score_result = metrics.f1_score(y_test, y_prediction, average='weighted')
        return accuracy_result, f1_score_result
    else:
        return np.nan, np.nan


def generate_logistic_regression_and_evaluate(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]

    if len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        logisticRegr = LogisticRegression(max_iter=1000)
        logisticRegr.fit(X_train, y_train)
        y_prediction = logisticRegr.predict(X_test)

        accuracy_result = metrics.accuracy_score(y_test, y_prediction)
        f1_score_result = metrics.f1_score(y_test, y_prediction, average='weighted')
        return accuracy_result, f1_score_result
    else:
        return np.nan, np.nan


def generate_random_forest_and_evaluate(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]

    if len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_prediction = rf.predict(X_test)

        accuracy_result = metrics.accuracy_score(y_test, y_prediction)
        f1_score_result = metrics.f1_score(y_test, y_prediction, average='weighted')
        return accuracy_result, f1_score_result
    else:
        return np.nan, np.nan


def map_categorical_fields_to_numerical_values(df, columns, mapping):
    for column in columns:
        df[column] = df[column].map(mapping[column])
    return df


def get_mapping_for_categorical_fields_to_numerical(df, columns):
    category_mapping_for_all_columns = dict()
    inverted_category_mapping_for_all_columns = dict()

    for column in columns:
        unique_categories = df[column].unique()
        category_mapping = dict()
        for i in range(len(unique_categories)):
            category_mapping[unique_categories[i]] = i + 1
        category_mapping_for_all_columns[column] = category_mapping

    for column in columns:
        inverted_category_mapping_for_all_columns[column] = dict((v, k) for k, v in category_mapping_for_all_columns[column].items())

    return category_mapping_for_all_columns, inverted_category_mapping_for_all_columns


def convert_specific_columns(df, input_file):
    input_file = str(input_file)
    if re.search('.*Faker_Elektronischer_Impfpass_Output.csv', input_file):
        df['Tetanus und Diphterie Impfung'] = df['Tetanus und Diphterie Impfung'].map({1: 'True', 0: 'False', True: 'True', False: 'False', 'True': 'True', 'False': 'False'})
        df['Polio Impfung'] = df['Polio Impfung'].map({1: 'True', 0: 'False', True: 'True', False: 'False', 'True': 'True', 'False': 'False'})
        df['FSME Impfung'] = df['FSME Impfung'].map({1: 'True', 0: 'False', True: 'True', False: 'False', 'True': 'True', 'False': 'False'})
        df['Covid Impfung'] = df['Covid Impfung'].map({1: 'True', 0: 'False', True: 'True', False: 'False', 'True': 'True', 'False': 'False'})

        df['Geburtsdatum'] = pd.to_datetime(df['Geburtsdatum'], format='%Y-%m-%d')
        df['Tetanus und Diphterie Impfdatum'] = pd.to_datetime(df['Tetanus und Diphterie Impfdatum'], format='%Y-%m-%d')
        df['Tetanus und Diphterie Ablaufdatum des Impfstoffs'] = pd.to_datetime(df['Tetanus und Diphterie Ablaufdatum des Impfstoffs'], format='%Y-%m-%d')
        df['Polio Impfdatum'] = pd.to_datetime(df['Polio Impfdatum'], format='%Y-%m-%d')
        df['Polio Ablaufdatum des Impfstoffs'] = pd.to_datetime(df['Polio Ablaufdatum des Impfstoffs'], format='%Y-%m-%d')
        df['FSME Impfdatum'] = pd.to_datetime(df['FSME Impfdatum'], format='%Y-%m-%d')
        df['FSME Ablaufdatum des Impfstoffs'] = pd.to_datetime(df['FSME Ablaufdatum des Impfstoffs'], format='%Y-%m-%d')
        df['Covid Impfdatum'] = pd.to_datetime(df['Covid Impfdatum'], format='%Y-%m-%d')
        df['Covid Ablaufdatum des Impfstoffs'] = pd.to_datetime(df['Covid Ablaufdatum des Impfstoffs'], format='%Y-%m-%d')
        return df
    return df


def add_modified_date_column(df, csv_file_name):
    csv_file_name = str(csv_file_name)
    if re.search('.*{}'.format('Faker_Elektronischer_Impfpass_Output.csv'), csv_file_name):
        df['Geburtsdatum_year'] = [d.year if d is not pd.NaT else pd.NaT for d in df['Geburtsdatum']]
    return df


def main():
    stdout_backup = sys.stdout
    sys.stdout = UnbufferedLogging(sys.stdout)

    check_if_required_dirs_exist()
    csv_files = get_list_of_input_csv()

    counter = 0
    for csv_file in csv_files:
        anonymous_df = anonymize_data(csv_file)
        check_applicability_for_csv(anonymous_df, csv_file, counter)
        counter += 1

    sys.stdout = stdout_backup
    log_file_handler.close()


if __name__ == "__main__":
    main()
