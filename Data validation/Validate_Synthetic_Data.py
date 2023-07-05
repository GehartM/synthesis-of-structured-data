import sdmetrics.single_column
from sdmetrics.single_table import GMLogLikelihood
import pandas as pd
from collections import defaultdict
import os
import sys
import csv
from pathlib import Path
import re
import matplotlib.pyplot as plt
from dython import nominal
import calendar
import numpy as np
from sklearn import tree, svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import datetime


original_input_folder = Path('Input_Original')
synthetic_input_folder = Path('Input_Synthetic')
output_folder = Path('Output')

original_synthetic_file_mapping = defaultdict(list)

log_file_handler = open('validate_synthetic_data_{}.log'.format(str(datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))), 'w')


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
    stop_script = False
    check_if_dir_exists(output_folder)
    if not check_if_dir_exists(original_input_folder):
        stop_script = True
        print('\t[!] Provide the original csv-files inside the newly created "{}"-folder.\n\t[!] Afterwards start the script again.'.format(original_input_folder))
    if not check_if_dir_exists(synthetic_input_folder):
        stop_script = True
        print('\t[!] Provide the synthetic csv-files inside the newly created "{}"-folder.\n\t[!] Afterwards start the script again.'.format(synthetic_input_folder))
    if stop_script:
        sys.exit(1)


def check_if_dir_exists(folder_name):
    if not os.path.exists(folder_name):
        print('\n\t[-] "{}"-folder was created.'.format(folder_name))
        os.makedirs(folder_name)
        return False
    return True


def check_if_file_exists(file_name):
    if not os.path.exists(file_name):
        return False
    return True


def check_if_synthetic_file_exists_regex(file_name_regex, csv_file):
    found_synthetic_file = False
    for file_name in os.listdir(synthetic_input_folder):
        if re.search(file_name_regex, file_name):
            original_synthetic_file_mapping[csv_file].append(file_name)
            found_synthetic_file = True
    return found_synthetic_file


def check_if_synthetic_files_exists(csv_files):
    missing_synthetic_file = False
    for csv_file in csv_files:
        if not check_if_synthetic_file_exists_regex('.*_{}'.format(csv_file), csv_file):
            print('\n\t[!] Missing the synthetic csv-file for "{}"!\n\t[!] Provide the synthetic csv-file inside "{}".\n\t[!] Afterwards start the script again.'.format(csv_file, synthetic_input_folder))
            missing_synthetic_file = True
    return ~missing_synthetic_file


def get_list_of_input_csv():
    print('\n\t[-] Scanning "{}"-folder for csv-files...'.format(original_input_folder))
    files_in_dir = os.listdir(original_input_folder)
    csv_files = list(filter(lambda f: f.endswith('.csv'), files_in_dir))
    if len(csv_files) > 0:
        for csv_file in csv_files:
            print('\t\t[.] Found "{}"'.format(csv_file))
    else:
        print('\t[!] No csv-files were found inside the folder "{}"!'.format(original_input_folder))
        sys.exit(1)

    if not check_if_synthetic_files_exists(csv_files):
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
    delimiter = get_delimiter_of_csv(input_file)
    if delimiter is None:
        print('\n\t[!] The delimiter for "{}" could not be automatically determined!'.format(input_file))
        while delimiter is None:
            delimiter = input('\tEnter the delimiter for this file: ')
            if len(delimiter) != 1:
                print('\t[!] The delimiter can only be one character!')
                delimiter = None

    try:
        df = pd.read_csv(input_file, sep=delimiter, decimal=',')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, sep=delimiter, decimal=',', encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, sep=delimiter, decimal=',', encoding='ISO-8859-1')

    input_file_str = str(input_file)
    if re.search('WGAN_.*', input_file_str) or re.search('WGAN-GP_.*', input_file_str):
        df = join_splitted_datetime_columns_together(input_file_str, df)
        df = revert_masked_values_to_nan(input_file_str, df)

    df = convert_specific_columns(input_file, df)
    return add_modified_date_column(df, input_file)


def revert_masked_values_to_nan(input_file, df):
    if re.search('.*Case_Information.csv', input_file):
        df['province'] = df['province'].replace(['Not known'], np.nan)
        df['muni_city'] = df['muni_city'].replace(['Not known'], np.nan)
        df['region'] = df['region'].replace(['Not known'], np.nan)
        df['home_quarantined'] = df['home_quarantined'].replace(['Not known'], np.nan)
        df['pregnant'] = df['pregnant'].replace(['Not known'], np.nan)
        return df
    elif re.search('.*Faker_Elektronischer_Impfpass_Output.csv', input_file):
        df['Tetanus und Diphterie Impfstoff'] = df['Tetanus und Diphterie Impfstoff'].replace([0], np.nan)
        df['Tetanus und Diphterie Impfstoffhersteller'] = df['Tetanus und Diphterie Impfstoffhersteller'].replace([0], np.nan)
        df['Polio Impfstoff'] = df['Polio Impfstoff'].replace([0], np.nan)
        df['Polio Impfstoffhersteller'] = df['Polio Impfstoffhersteller'].replace([0], np.nan)
        df['FSME Impfstoff'] = df['FSME Impfstoff'].replace([0], np.nan)
        df['FSME Impfstoffhersteller'] = df['FSME Impfstoffhersteller'].replace([0], np.nan)
        df['Covid Impfstoff'] = df['Covid Impfstoff'].replace([0], np.nan)
        df['Covid Impfstoffhersteller'] = df['Covid Impfstoffhersteller'].replace([0], np.nan)
        return df
    elif re.search('.*Online Retail.csv', input_file):
        df['CustomerID'] = df['CustomerID'].replace([0], np.nan)
        return df
    else:
        return df


def join_splitted_datetime_columns_together(input_file, df):
    if re.search('.*Case_Information.csv', input_file):
        df['date_announced'] = df.apply(join_to_date, day='date_announced_day', month='date_announced_month', year='date_announced_year', date_format='%d.%m.%Y', axis=1)
        df['date_recovered'] = df.apply(join_to_date, day='date_recovered_day', month='date_recovered_month', year='date_recovered_year', date_format='%d.%m.%Y', axis=1)
        df['date_of_death'] = df.apply(join_to_date, day='date_of_death_day', month='date_of_death_month', year='date_of_death_year', date_format='%d.%m.%Y', axis=1)
        df['date_announced_as_removed'] = df.apply(join_to_date, day='date_announced_as_removed_day', month='date_announced_as_removed_month', year='date_announced_as_removed_year', date_format='%d.%m.%Y', axis=1)
        df['date_of_onset_of_symptoms'] = df.apply(join_to_date, day='date_of_onset_of_symptoms_day', month='date_of_onset_of_symptoms_month', year='date_of_onset_of_symptoms_year', date_format='%d.%m.%Y', axis=1)
        df.drop('date_announced_day', inplace=True, axis=1)
        df.drop('date_announced_month', inplace=True, axis=1)
        df.drop('date_announced_year', inplace=True, axis=1)
        df.drop('date_recovered_day', inplace=True, axis=1)
        df.drop('date_recovered_month', inplace=True, axis=1)
        df.drop('date_recovered_year', inplace=True, axis=1)
        df.drop('date_of_death_day', inplace=True, axis=1)
        df.drop('date_of_death_month', inplace=True, axis=1)
        df.drop('date_of_death_year', inplace=True, axis=1)
        df.drop('date_announced_as_removed_day', inplace=True, axis=1)
        df.drop('date_announced_as_removed_month', inplace=True, axis=1)
        df.drop('date_announced_as_removed_year', inplace=True, axis=1)
        df.drop('date_of_onset_of_symptoms_day', inplace=True, axis=1)
        df.drop('date_of_onset_of_symptoms_month', inplace=True, axis=1)
        df.drop('date_of_onset_of_symptoms_year', inplace=True, axis=1)
        return df
    elif re.search('.*Faker_Elektronischer_Impfpass_Output.csv', input_file):
        df['Geburtsdatum'] = df.apply(join_to_date, day='Geburtsdatum_Tag', month='Geburtsdatum_Monat', year='Geburtsdatum_Jahr', date_format='%Y-%m-%d', axis=1)
        df['Tetanus und Diphterie Impfdatum'] = df.apply(join_to_date, day='Tetanus und Diphterie Impfdatum_Tag', month='Tetanus und Diphterie Impfdatum_Monat', year='Tetanus und Diphterie Impfdatum_Jahr', date_format='%Y-%m-%d', axis=1)
        df['Tetanus und Diphterie Ablaufdatum des Impfstoffs'] = df.apply(join_to_date, day='Tetanus und Diphterie Ablaufdatum des Impfstoffs_Tag', month='Tetanus und Diphterie Ablaufdatum des Impfstoffs_Monat', year='Tetanus und Diphterie Ablaufdatum des Impfstoffs_Jahr', date_format='%Y-%m-%d', axis=1)
        df['Polio Impfdatum'] = df.apply(join_to_date, day='Polio Impfdatum_Tag', month='Polio Impfdatum_Monat', year='Polio Impfdatum_Jahr', date_format='%Y-%m-%d', axis=1)
        df['Polio Ablaufdatum des Impfstoffs'] = df.apply(join_to_date, day='Polio Ablaufdatum des Impfstoffs_Tag', month='Polio Ablaufdatum des Impfstoffs_Monat', year='Polio Ablaufdatum des Impfstoffs_Jahr', date_format='%Y-%m-%d', axis=1)
        df['FSME Impfdatum'] = df.apply(join_to_date, day='FSME Impfdatum_Tag', month='FSME Impfdatum_Monat', year='FSME Impfdatum_Jahr', date_format='%Y-%m-%d', axis=1)
        df['FSME Ablaufdatum des Impfstoffs'] = df.apply(join_to_date, day='FSME Ablaufdatum des Impfstoffs_Tag', month='FSME Ablaufdatum des Impfstoffs_Monat', year='FSME Ablaufdatum des Impfstoffs_Jahr', date_format='%Y-%m-%d', axis=1)
        df['Covid Impfdatum'] = df.apply(join_to_date, day='Covid Impfdatum_Tag', month='Covid Impfdatum_Monat', year='Covid Impfdatum_Jahr', date_format='%Y-%m-%d', axis=1)
        df['Covid Ablaufdatum des Impfstoffs'] = df.apply(join_to_date, day='Covid Ablaufdatum des Impfstoffs_Tag', month='Covid Ablaufdatum des Impfstoffs_Monat', year='Covid Ablaufdatum des Impfstoffs_Jahr', date_format='%Y-%m-%d', axis=1)
        df.drop('Geburtsdatum_Tag', inplace=True, axis=1)
        df.drop('Geburtsdatum_Monat', inplace=True, axis=1)
        df.drop('Geburtsdatum_Jahr', inplace=True, axis=1)
        df.drop('Tetanus und Diphterie Impfdatum_Tag', inplace=True, axis=1)
        df.drop('Tetanus und Diphterie Impfdatum_Monat', inplace=True, axis=1)
        df.drop('Tetanus und Diphterie Impfdatum_Jahr', inplace=True, axis=1)
        df.drop('Tetanus und Diphterie Ablaufdatum des Impfstoffs_Tag', inplace=True, axis=1)
        df.drop('Tetanus und Diphterie Ablaufdatum des Impfstoffs_Monat', inplace=True, axis=1)
        df.drop('Tetanus und Diphterie Ablaufdatum des Impfstoffs_Jahr', inplace=True, axis=1)
        df.drop('Polio Impfdatum_Tag', inplace=True, axis=1)
        df.drop('Polio Impfdatum_Monat', inplace=True, axis=1)
        df.drop('Polio Impfdatum_Jahr', inplace=True, axis=1)
        df.drop('Polio Ablaufdatum des Impfstoffs_Tag', inplace=True, axis=1)
        df.drop('Polio Ablaufdatum des Impfstoffs_Monat', inplace=True, axis=1)
        df.drop('Polio Ablaufdatum des Impfstoffs_Jahr', inplace=True, axis=1)
        df.drop('FSME Impfdatum_Tag', inplace=True, axis=1)
        df.drop('FSME Impfdatum_Monat', inplace=True, axis=1)
        df.drop('FSME Impfdatum_Jahr', inplace=True, axis=1)
        df.drop('FSME Ablaufdatum des Impfstoffs_Tag', inplace=True, axis=1)
        df.drop('FSME Ablaufdatum des Impfstoffs_Monat', inplace=True, axis=1)
        df.drop('FSME Ablaufdatum des Impfstoffs_Jahr', inplace=True, axis=1)
        df.drop('Covid Impfdatum_Tag', inplace=True, axis=1)
        df.drop('Covid Impfdatum_Monat', inplace=True, axis=1)
        df.drop('Covid Impfdatum_Jahr', inplace=True, axis=1)
        df.drop('Covid Ablaufdatum des Impfstoffs_Tag', inplace=True, axis=1)
        df.drop('Covid Ablaufdatum des Impfstoffs_Monat', inplace=True, axis=1)
        df.drop('Covid Ablaufdatum des Impfstoffs_Jahr', inplace=True, axis=1)
        return df
    elif re.search('.*Faker_Jö_Bonusclub_Output.csv', input_file):
        df['Kaufdatum'] = df.apply(join_to_datetime, day='Kaufdatum_Tag', month='Kaufdatum_Monat', year='Kaufdatum_Jahr', hours='Kaufdatum_Stunde', minutes='Kaufdatum_Minute', seconds='Kaufdatum_Sekunde', date_format='%Y-%m-%d %H:%M:%S', axis=1)
        df['Geburtsdatum'] = df.apply(join_to_date, day='Geburtsdatum_Tag', month='Geburtsdatum_Monat', year='Geburtsdatum_Jahr', date_format='%Y-%m-%d', axis=1)
        df.drop('Kaufdatum_Tag', inplace=True, axis=1)
        df.drop('Kaufdatum_Monat', inplace=True, axis=1)
        df.drop('Kaufdatum_Jahr', inplace=True, axis=1)
        df.drop('Kaufdatum_Stunde', inplace=True, axis=1)
        df.drop('Kaufdatum_Minute', inplace=True, axis=1)
        df.drop('Kaufdatum_Sekunde', inplace=True, axis=1)
        df.drop('Geburtsdatum_Tag', inplace=True, axis=1)
        df.drop('Geburtsdatum_Monat', inplace=True, axis=1)
        df.drop('Geburtsdatum_Jahr', inplace=True, axis=1)
        return df
    elif re.search('.*Online Retail.csv', input_file):
        df['InvoiceDate'] = df.apply(join_to_datetime, day='InvoiceDate_day', month='InvoiceDate_month',
                                   year='InvoiceDate_year', hours='InvoiceDate_hour', minutes='InvoiceDate_minute',
                                   seconds=0, date_format='%d.%m.%Y %H:%M', ignore_seconds=True, axis=1)
        df.drop('InvoiceDate_day', inplace=True, axis=1)
        df.drop('InvoiceDate_month', inplace=True, axis=1)
        df.drop('InvoiceDate_year', inplace=True, axis=1)
        df.drop('InvoiceDate_hour', inplace=True, axis=1)
        df.drop('InvoiceDate_minute', inplace=True, axis=1)
        return df
    else:
        print('\n\t[!] Got unsupported csv-file "{}"!'.format(input_file))
        sys.exit(1)


def join_to_datetime(row, day, month, year, hours, minutes, seconds, date_format, ignore_seconds=False):
    day = int(row[day])
    month = int(row[month])
    year = int(row[year])
    hours = int(row[hours])
    minutes = int(row[minutes])
    if not ignore_seconds:
        seconds = int(row[seconds])
    else:
        seconds = 0

    if day < 1 or day > 31 or month < 1 or month > 12 or year < 1000 or hours < 0 or hours > 23 or minutes < 0 or minutes > 59 or seconds < 0 or seconds > 59:
        return np.nan

    try:
        if ignore_seconds:
            joined_datetime = datetime.datetime(year, month, day, hours, minutes, 0)
        else:
            joined_datetime = datetime.datetime(year, month, day, hours, minutes, seconds)
        joined_datetime = str(joined_datetime.strftime(date_format))
    except(ValueError):
        joined_datetime = np.nan

    return joined_datetime


def join_to_date(row, day, month, year, date_format):
    day = int(row[day])
    month = int(row[month])
    year = int(row[year])

    if day < 1 or day > 31 or month < 1 or month > 12 or year < 1000:
        return np.nan

    try:
        joined_datetime = datetime.date(year, month, day)
        joined_datetime = str(joined_datetime.strftime(date_format))
    except(ValueError):
        joined_datetime = np.nan

    return joined_datetime


def convert_specific_columns(input_file, df):
    input_file = str(input_file)
    if re.search('.*Case_Information.csv', input_file):
        df['home_quarantined'] = df['home_quarantined'].map({'Yes': 'True', 'No': 'False', True: 'True', False: 'False', 'True': 'True', 'False': 'False'})
        df['pregnant'] = df['pregnant'].map({'Yes': 'True', 'No': 'False', True: 'True', False: 'False', 'True': 'True', 'False': 'False'})
        df['date_announced'] = pd.to_datetime(df['date_announced'], format='%d.%m.%Y')
        df['date_recovered'] = pd.to_datetime(df['date_recovered'], format='%d.%m.%Y')
        df['date_of_death'] = pd.to_datetime(df['date_of_death'], format='%d.%m.%Y')
        df['date_announced_as_removed'] = pd.to_datetime(df['date_announced_as_removed'], format='%d.%m.%Y')
        df['date_of_onset_of_symptoms'] = pd.to_datetime(df['date_of_onset_of_symptoms'], format='%d.%m.%Y')
        return df
    elif re.search('.*Faker_Elektronischer_Impfpass_Output.csv', input_file):
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
    elif re.search('.*Faker_Jö_Bonusclub_Output.csv', input_file):
        df['Kaufdatum'] = pd.to_datetime(df['Kaufdatum'], format='%Y-%m-%d %H:%M:%S')
        df['Geburtsdatum'] = pd.to_datetime(df['Geburtsdatum'], format='%Y-%m-%d')
        return df
    elif re.search('.*Online Retail.csv', input_file):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d.%m.%Y %H:%M')
        return df

    return df


def get_structural_similarity(original_df, synthetic_df, csv_file_name, synthetic_method, datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column):
    df = pd.DataFrame()

    df['CSV-Datei'] = [csv_file_name]
    df['Methode'] = [synthetic_method]
    df.set_index('CSV-Datei', inplace=True)

    # Check mean, median and standard deviation
    # best 1.0, worst 0.0
    # https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/statisticsimilarity
    for column in datetime_columns + continuous_columns:
        df['Mittelwert_Num_{}'.format(column)] = [statisical_similarity(original_df, synthetic_df, column, 'mean')]
        df['Median_Num_{}'.format(column)] = [statisical_similarity(original_df, synthetic_df, column, 'median')]
        df['Standardabweichung_Num_{}'.format(column)] = [statisical_similarity(original_df, synthetic_df, column, 'std')]

    # Check if date and numerical ranges are represented
    # best 1.0, worst 0.0
    # https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/rangecoverage
    for column in datetime_columns + continuous_columns:
        df['Werteumfang_Num_{}'.format(column)] = [range_coverage(original_df, synthetic_df, column)]

    # Check if date and numbers are inside the original boundaries
    # best 1.0, worst 0.0
    # https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/boundaryadherence

    for column in datetime_columns + continuous_columns:
        df['Wertebereich_min_max_Num_{}'.format(column)] = [boundary_adherence(original_df, synthetic_df, column)]

    # Check if all categories are represented
    # best 1.0, worst 0.0
    # https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/categorycoverage
    for column in categorical_columns + bool_columns:
        df['Werteumfang_Kat_{}'.format(column)] = [category_coverage(original_df, synthetic_df, column)]

    # Check the frequency of each category
    # best 1.0, worst 0.0
    # https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/tvcomplement
    for column in datetime_columns + continuous_columns:
        df['Häufigkeit_Kat_{}'.format(column)] = [category_frequency(original_df, synthetic_df, column)]

    # Check the proportion of missing values
    # best 1.0, worst 0.0
    # https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/missingvaluesimilarity
    for column in datetime_columns + continuous_columns + categorical_columns + bool_columns:
        df['Fehlende_Werte_{}'.format(column)] = [missing_value_proportion(original_df, synthetic_df, column)]

    return df


def get_synthetic_bivariate_accuracy(original_df, synthetic_df, csv_file_name, synthetic_method, datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column):
    csv_file_name = str(csv_file_name)
    output_file_name = 'Validation_result_2_bivariate_accuracy_{}_{}.xlsx'.format(synthetic_method, csv_file_name[:-4])
    if re.search('.*{}'.format('Case_Information.csv'), csv_file_name):
        original_df['Alter_Kategorie'] = pd.cut(x=original_df['age'], bins=[-1, 14, 29, 44, 59, 74, 150], labels=['<15', '15-29', '30-44', '45-59', '60-74', '>74'])
        synthetic_df['Alter_Kategorie'] = pd.cut(x=synthetic_df['age'], bins=[-1, 14, 29, 44, 59, 74, 150], labels=['<15', '15-29', '30-44', '45-59', '60-74', '>74'])

        health_status_per_age_original = pd.crosstab(original_df['Alter_Kategorie'], original_df['health_status'])
        health_status_per_age_synthetic = pd.crosstab(synthetic_df['Alter_Kategorie'], synthetic_df['health_status'])
        write_df_to_excel(health_status_per_age_original, output_file_name, 'o_hea_age', execution_counter=0)
        write_df_to_excel(health_status_per_age_synthetic, output_file_name, 's_hea_age')

        status_per_region_and_sex_original = pd.crosstab(original_df['region'], [original_df['status'], original_df['sex']])
        status_per_region_and_sex_synthetic = pd.crosstab(synthetic_df['region'], [synthetic_df['status'], synthetic_df['sex']])
        write_df_to_excel(status_per_region_and_sex_original, output_file_name, 'o_stat_reg')
        write_df_to_excel(status_per_region_and_sex_synthetic, output_file_name, 's_stat_reg')
    elif re.search('.*{}'.format('Faker_Jö_Bonusclub_Output.csv'), csv_file_name):
        original_df.replace({pd.NaT: 0}, inplace=True)
        synthetic_df.replace({pd.NaT: 0}, inplace=True)
        original_df['Alter'] = 2023 - original_df['Geburtsdatum_year']
        synthetic_df['Alter'] = 2023 - synthetic_df['Geburtsdatum_year']
        original_df['Alter_Kategorie'] = pd.cut(x=original_df['Alter'], bins=[-1, 14, 29, 44, 59, 74, 150, 3000], labels=['<15', '15-29', '30-44', '45-59', '60-74', '>74', 'NA'])
        synthetic_df['Alter_Kategorie'] = pd.cut(x=synthetic_df['Alter'], bins=[-1, 14, 29, 44, 59, 74, 150, 3000], labels=['<15', '15-29', '30-44', '45-59', '60-74', '>74', 'NA'])

        bought_items_per_age_original = pd.crosstab(original_df['Alter_Kategorie'], original_df['Produktname'])
        bought_items_per_age_synthetic = pd.crosstab(synthetic_df['Alter_Kategorie'], synthetic_df['Produktname'])
        write_df_to_excel(bought_items_per_age_original, output_file_name, 'o_prod_age', execution_counter=0)
        write_df_to_excel(bought_items_per_age_synthetic, output_file_name, 's_prod_age')

        bought_kategorie_per_age_original = pd.crosstab(original_df['Alter_Kategorie'], original_df['Kategorie'])
        bought_kategorie_per_age_synthetic = pd.crosstab(synthetic_df['Alter_Kategorie'], synthetic_df['Kategorie'])
        write_df_to_excel(bought_kategorie_per_age_original, output_file_name, 'o_kat_age')
        write_df_to_excel(bought_kategorie_per_age_synthetic, output_file_name, 's_kat_age')
    elif re.search('.*{}'.format('Online Retail.csv'), csv_file_name):
        item_per_country_original = pd.crosstab(original_df['Description'], original_df['Country'])
        item_per_country_synthetic = pd.crosstab(synthetic_df['Description'], synthetic_df['Country'])
        write_df_to_excel(item_per_country_original, output_file_name, 'o_desc_country', execution_counter=0)
        write_df_to_excel(item_per_country_synthetic, output_file_name, 's_desc_country')

        original_df['Canceled_order'] = np.where(original_df['InvoiceNo'].str.startswith('C'), True, False)
        synthetic_df['Canceled_order'] = np.where(synthetic_df['InvoiceNo'].str.startswith('C'), True, False)
        item_cancelation_rate_original = pd.crosstab(original_df['Description'], original_df['Canceled_order'])
        item_cancelation_rate_synthetic = pd.crosstab(synthetic_df['Description'], synthetic_df['Canceled_order'])
        write_df_to_excel(item_cancelation_rate_original, output_file_name, 'o_desc_canc')
        write_df_to_excel(item_cancelation_rate_synthetic, output_file_name, 's_desc_canc')

        item_per_month_original = pd.crosstab(original_df['Description'], original_df['InvoiceDate_month'])
        item_per_month_synthetic = pd.crosstab(synthetic_df['Description'], synthetic_df['InvoiceDate_month'])
        write_df_to_excel(item_per_month_original, output_file_name, 'o_desc_month')
        write_df_to_excel(item_per_month_synthetic, output_file_name, 's_desc_month')
    elif re.search('.*{}'.format('Faker_Elektronischer_Impfpass_Output.csv'), csv_file_name):
        original_df.replace({pd.NaT: 0}, inplace=True)
        synthetic_df.replace({pd.NaT: 0}, inplace=True)

        original_df['Alter'] = 2023 - original_df['Geburtsdatum_year']
        synthetic_df['Alter'] = 2023 - synthetic_df['Geburtsdatum_year']
        original_df['Alter_Kategorie'] = pd.cut(x=original_df['Alter'], bins=[-1, 14, 29, 44, 59, 74, 150, 3000], labels=['<15', '15-29', '30-44', '45-59', '60-74', '>74', 'NA'])
        synthetic_df['Alter_Kategorie'] = pd.cut(x=synthetic_df['Alter'], bins=[-1, 14, 29, 44, 59, 74, 150, 3000], labels=['<15', '15-29', '30-44', '45-59', '60-74', '>74', 'NA'])


        vaccination_per_age_original = pd.crosstab(original_df['Alter_Kategorie'], original_df['Tetanus und Diphterie Impfung'])
        vaccination_per_age_synthetic = pd.crosstab(synthetic_df['Alter_Kategorie'], synthetic_df['Tetanus und Diphterie Impfung'])
        write_df_to_excel(vaccination_per_age_original, output_file_name, 'o_tet_age', execution_counter=0)
        write_df_to_excel(vaccination_per_age_synthetic, output_file_name, 's_tet_age')

        vaccination_per_age_original = pd.crosstab(original_df['Alter_Kategorie'], original_df['Polio Impfung'])
        vaccination_per_age_synthetic = pd.crosstab(synthetic_df['Alter_Kategorie'], synthetic_df['Polio Impfung'])
        write_df_to_excel(vaccination_per_age_original, output_file_name, 'o_polio_age')
        write_df_to_excel(vaccination_per_age_synthetic, output_file_name, 's_polio_age')

        vaccination_per_age_original = pd.crosstab(original_df['Alter_Kategorie'], original_df['FSME Impfung'])
        vaccination_per_age_synthetic = pd.crosstab(synthetic_df['Alter_Kategorie'], synthetic_df['FSME Impfung'])
        write_df_to_excel(vaccination_per_age_original, output_file_name, 'o_fsme_age')
        write_df_to_excel(vaccination_per_age_synthetic, output_file_name, 's_fsme_age')

        vaccination_per_age_original = pd.crosstab(original_df['Alter_Kategorie'], original_df['Covid Impfung'])
        vaccination_per_age_synthetic = pd.crosstab(synthetic_df['Alter_Kategorie'], synthetic_df['Covid Impfung'])
        write_df_to_excel(vaccination_per_age_original, output_file_name, 'o_cov_age')
        write_df_to_excel(vaccination_per_age_synthetic, output_file_name, 's_cov_age')



        vaccination_per_gender_original = pd.crosstab(original_df['Geschlecht'], original_df['Tetanus und Diphterie Impfung'])
        vaccination_per_gender_synthetic = pd.crosstab(synthetic_df['Geschlecht'], synthetic_df['Tetanus und Diphterie Impfung'])
        write_df_to_excel(vaccination_per_gender_original, output_file_name, 'o_tet_gen')
        write_df_to_excel(vaccination_per_gender_synthetic, output_file_name, 's_tet_gen')

        vaccination_per_gender_original = pd.crosstab(original_df['Geschlecht'], original_df['Polio Impfung'])
        vaccination_per_gender_synthetic = pd.crosstab(synthetic_df['Geschlecht'], synthetic_df['Polio Impfung'])
        write_df_to_excel(vaccination_per_gender_original, output_file_name, 'o_polio_gen')
        write_df_to_excel(vaccination_per_gender_synthetic, output_file_name, 's_polio_gen')

        vaccination_per_gender_original = pd.crosstab(original_df['Geschlecht'], original_df['FSME Impfung'])
        vaccination_per_gender_synthetic = pd.crosstab(synthetic_df['Geschlecht'], synthetic_df['FSME Impfung'])
        write_df_to_excel(vaccination_per_gender_original, output_file_name, 'o_fsme_gen')
        write_df_to_excel(vaccination_per_gender_synthetic, output_file_name, 's_fsme_gen')

        vaccination_per_gender_original = pd.crosstab(original_df['Geschlecht'], original_df['Covid Impfung'])
        vaccination_per_gender_synthetic = pd.crosstab(synthetic_df['Geschlecht'], synthetic_df['Covid Impfung'])
        write_df_to_excel(vaccination_per_gender_original, output_file_name, 'o_cov_gen')
        write_df_to_excel(vaccination_per_gender_synthetic, output_file_name, 's_cov_gen')
    else:
        print('\n\t[!] Got unsupported csv-file "{}"!'.format(csv_file_name))
        sys.exit(1)


def check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, column):
    if type(column) is list:
        if len(column) == 2:
            column_text = '+'.join(column)
            original_df['combination'] = original_df[column[0]].astype(str) + original_df[column[1]].astype(str)
            synthetic_df['combination'] = synthetic_df[column[0]].astype(str) + synthetic_df[column[1]].astype(str)
            synthetic_df['{}_in_Original'.format(column_text)] = synthetic_df['combination'].isin(original_df['combination'])
            synthetic_df.drop('combination', inplace=True, axis=1)
        else:
            print('\n\t[!] Got invalid number of list item. Expected 2, got {}!'.format(len(column)))
            sys.exit(1)
    else:
        column_text = column
        synthetic_df['{}_in_Original'.format(column_text)] = synthetic_df[column].isin(original_df[column])

    if synthetic_df['{}_in_Original'.format(column_text)].any():
        pii_presence = synthetic_df['{}_in_Original'.format(column_text)].value_counts().loc[True]
        if pii_presence > 0:
            print('\t\t[-] Found {} identical pii information inside "{}" of "{}_{}".'.format(pii_presence, column_text, synthetic_method, csv_file_name))
    return synthetic_df


def check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, columns):
    original_df['combination'] = ''
    synthetic_df['combination'] = ''
    column_text = '+'.join(columns)
    for column in columns:
        original_df['combination'] = original_df['combination'] + original_df[column].astype(str)
        synthetic_df['combination'] = synthetic_df['combination'] + synthetic_df[column].astype(str)
    synthetic_df['{}_in_Original'.format(column_text)] = synthetic_df['combination'].isin(original_df['combination'])
    synthetic_df.drop('combination', inplace=True, axis=1)

    if synthetic_df['{}_in_Original'.format(column_text)].any():
        pii_presence = synthetic_df['{}_in_Original'.format(column_text)].value_counts().loc[True]
        if pii_presence > 0:
            print('\t\t[-] Found {} identical quasi identifiers for {} in "{}_{}".'.format(pii_presence, column_text, synthetic_method, csv_file_name))
    return synthetic_df


def check_synthetic_privacy(original_df, synthetic_df, csv_file_name, synthetic_method, datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column):
    backup_synthetic_df = synthetic_df
    csv_file_name = str(csv_file_name)
    output_file_name = 'Validation_result_5_privacy_{}_{}.xlsx'.format(synthetic_method, csv_file_name[:-4])

    # Check for pii presence
    if re.search('.*{}'.format('Case_Information.csv'), csv_file_name):
        # There is no pii
        pass
    elif re.search('.*{}'.format('Faker_Jö_Bonusclub_Output.csv'), csv_file_name):
        synthetic_df = check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, 'Mitgliedsnummer')
        synthetic_df = check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname'])
        synthetic_df = check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, 'Straße')
        synthetic_df = check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, 'Telefonnummer')
        synthetic_df = check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, 'E-Mailadresse')
        write_df_to_excel(synthetic_df, output_file_name, 'PII', execution_counter=0)
    elif re.search('.*{}'.format('Online Retail.csv'), csv_file_name):
        synthetic_df = check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, 'CustomerID')
        write_df_to_excel(synthetic_df, output_file_name, 'PII', execution_counter=0)
    elif re.search('.*{}'.format('Faker_Elektronischer_Impfpass_Output.csv'), csv_file_name):
        synthetic_df = check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname'])
        synthetic_df = check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, 'Sozialversicherungsnummer')
        synthetic_df = check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, ['Ort', 'Straße'])
        synthetic_df = check_for_pii_presence(original_df, synthetic_df, csv_file_name, synthetic_method, ['Postleitzahl', 'Straße'])
        write_df_to_excel(synthetic_df, output_file_name, 'PII', execution_counter=0)
    else:
        print('\n\t[!] Got unsupported csv-file "{}"!'.format(csv_file_name))
        sys.exit(1)


    # Check for presence of quasi identifiers
    synthetic_df = backup_synthetic_df
    if re.search('.*{}'.format('Case_Information.csv'), csv_file_name):
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['age', 'sex', 'pregnant', 'province', 'muni_city', 'region'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['age', 'sex', 'muni_city', 'pregnant'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['age', 'sex', 'muni_city'])
        write_df_to_excel(synthetic_df, output_file_name, 'Quasi_ID')
    elif re.search('.*{}'.format('Faker_Jö_Bonusclub_Output.csv'), csv_file_name):
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname', 'Straße', 'Geschlecht', 'Geburtsdatum_year', 'Postleitzahl'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname', 'Straße'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname', 'Geschlecht', 'Geburtsdatum_year', 'Postleitzahl'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname', 'Geburtsdatum_year', 'Postleitzahl'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname', 'Geburtsdatum_year'])
        #synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Geschlecht', 'Geburtsdatum_year', 'Postleitzahl'])
        write_df_to_excel(synthetic_df, output_file_name, 'Quasi_ID')
    elif re.search('.*{}'.format('Online Retail.csv'), csv_file_name):
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['CustomerID', 'Country', 'InvoiceDate_date'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['CustomerID', 'InvoiceDate_date'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Country', 'InvoiceDate_date'])
        write_df_to_excel(synthetic_df, output_file_name, 'Quasi_ID')
    elif re.search('.*{}'.format('Faker_Elektronischer_Impfpass_Output.csv'), csv_file_name):
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname', 'Straße', 'Ort', 'Geschlecht', 'Geburtsdatum_year', 'Postleitzahl'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname', 'Straße', 'Ort'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname', 'Geschlecht', 'Geburtsdatum_year', 'Postleitzahl'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname', 'Geburtsdatum_year', 'Postleitzahl'])
        synthetic_df = check_quasi_identifiers(original_df, synthetic_df, csv_file_name, synthetic_method, ['Vorname', 'Nachname', 'Geburtsdatum_year'])
        write_df_to_excel(synthetic_df, output_file_name, 'Quasi_ID')
    else:
        print('\n\t[!] Got unsupported csv-file "{}"!'.format(csv_file_name))
        sys.exit(1)


def check_applicability(original_df, synthetic_df, csv_file_name, synthetic_method, datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column):
    csv_file_name = str(csv_file_name)
    df = pd.DataFrame()

    df['CSV-Datei'] = [csv_file_name]
    df['Methode'] = [synthetic_method]
    df.set_index('CSV-Datei', inplace=True)

    if re.search('.*{}'.format('Case_Information.csv'), csv_file_name):
        # Calculate accuracy and f1 score using Decision Tree, Support Vector Machine, Logistic Regression and Random_Forest
        # 1.0 good prediction, 0.0 bad prediction
        feature_columns = ['age', 'sex', 'pregnant', 'region']
        target_column = 'status'
        original_df = original_df[original_df[target_column].isin(['Recovered', 'Died']) == True]
        synthetic_df = synthetic_df[synthetic_df[target_column].isin(['Recovered', 'Died']) == True]

        df = calculate_accuracy_and_f1_score_for_applicability(df, 'Accuracy_Original', 'F1-Score_Original', 'Accuracy_Synthetic', 'F1-Score_Synthetic', original_df, synthetic_df, categorical_columns, bool_columns, feature_columns, target_column, continuous_columns)
    elif re.search('.*{}'.format('Faker_Jö_Bonusclub_Output.csv'), csv_file_name):
        # Identify association rules of bought items using the Apriori algorithm
        # 1.0 all association rules could be preserved, 0.0 no association rules were preserved
        output_file_name = 'Validation_result_4_applicability_basket_analysis_{}_{}.xlsx'.format(synthetic_method, csv_file_name[:-4])
        assoc_rules_orig = basket_analysis(original_df, csv_file_name, synthetic_method, 'Mitgliedsnummer', 'Produktname', 'Anzahl', output_file_name, 'original', 0)
        assoc_rules_synth = basket_analysis(synthetic_df, csv_file_name, synthetic_method, 'Mitgliedsnummer', 'Produktname', 'Anzahl', output_file_name, 'synthetic')
        if assoc_rules_orig is None or assoc_rules_synth is None:
            df['Matched_association_rule_percentage'] = 0
        else:
            df['Matched_association_rule_percentage'] = get_percentage_of_similar_association_rules(assoc_rules_orig, assoc_rules_synth)
    elif re.search('.*{}'.format('Online Retail.csv'), csv_file_name):
        # Identify association rules of bought items using the Apriori algorithm
        # 1.0 all association rules could be preserved, 0.0 no association rules were preserved
        output_file_name = 'Validation_result_4_applicability_basket_analysis_{}_{}.xlsx'.format(synthetic_method, csv_file_name[:-4])
        assoc_rules_orig = basket_analysis(original_df, csv_file_name, synthetic_method, 'InvoiceNo', 'Description', 'Quantity', output_file_name, 'original', 0)
        assoc_rules_synth = basket_analysis(synthetic_df, csv_file_name, synthetic_method, 'InvoiceNo', 'Description', 'Quantity', output_file_name, 'synthetic')
        if assoc_rules_orig is None or assoc_rules_synth is None:
            df['Matched_association_rule_percentage'] = 0
        else:
            df['Matched_association_rule_percentage'] = get_percentage_of_similar_association_rules(assoc_rules_orig, assoc_rules_synth)
    elif re.search('.*{}'.format('Faker_Elektronischer_Impfpass_Output.csv'), csv_file_name):
        # Calculate accuracy and f1 score using Decision Tree, Support Vector Machine, Logistic Regression and Random_Forest
        # 1.0 good prediction, 0.0 bad prediction
        feature_columns = ['Geschlecht', 'Geburtsdatum_year']
        target_column = 'Tetanus und Diphterie Impfung'
        continuous_columns.append('Geburtsdatum_year')

        df = calculate_accuracy_and_f1_score_for_applicability(df, 'Accuracy_Original_Tet_Diph', 'F1-Score_Original_Tet_Diph', 'Accuracy_Synthetic_Tet_Diph', 'F1-Score_Synthetic_Tet_Diph', original_df, synthetic_df, categorical_columns, bool_columns, feature_columns, target_column, continuous_columns)

        target_column = 'Polio Impfung'
        df = calculate_accuracy_and_f1_score_for_applicability(df, 'Accuracy_Original_Polio', 'F1-Score_Original_Polio', 'Accuracy_Synthetic_Polio', 'F1-Score_Synthetic_Polio', original_df, synthetic_df, categorical_columns, bool_columns, feature_columns, target_column, continuous_columns)

        target_column = 'FSME Impfung'
        df = calculate_accuracy_and_f1_score_for_applicability(df, 'Accuracy_Original_FSME', 'F1-Score_Original_FSME', 'Accuracy_Synthetic_FSME', 'F1-Score_Synthetic_FSME', original_df, synthetic_df, categorical_columns, bool_columns, feature_columns, target_column, continuous_columns)

        target_column = 'Covid Impfung'
        df = calculate_accuracy_and_f1_score_for_applicability(df, 'Accuracy_Original_Cov', 'F1-Score_Original_Cov', 'Accuracy_Synthetic_Cov', 'F1-Score_Synthetic_Cov', original_df, synthetic_df, categorical_columns, bool_columns, feature_columns, target_column, continuous_columns)
    else:
        print('\n\t[!] Got unsupported csv-file "{}"!'.format(csv_file_name))
        sys.exit(1)
    return df


def encode_data_with_zero_and_one(value):
    if value >= 1:
        return 1
    else:
        return 0


def basket_analysis(df, csv_file_name, synthetic_method, invoice_number_column, item_name_column, item_quantity_column, output_file_name, type, execution_counter=1):
    min_support = 0.02
    if re.search('.*{}'.format('Faker_Jö_Bonusclub_Output.csv'), csv_file_name):
        if (synthetic_method == 'actGAN') and execution_counter != 0:
            min_support = 0.02
        else:
            min_support = 0.01

    df.drop(df[df[item_quantity_column] <= 0].index, inplace=True)
    bought_items = df.groupby([invoice_number_column, item_name_column])[item_quantity_column].sum().unstack().reset_index().fillna(0).set_index(invoice_number_column)
    bought_items = bought_items.applymap(encode_data_with_zero_and_one)
    bought_items = bought_items[(bought_items > 0).sum(axis=1) > 1]
    frequent_itemsets = apriori(bought_items, min_support=min_support, use_colnames=True).sort_values("support", ascending=False)
    if len(frequent_itemsets) == 0:
        print('\n\t[!] No items for basket analysis inside "{}"!'.format(csv_file_name))
        return None
    assoc_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.8).sort_values("lift", ascending=False).reset_index(drop=True)

    write_df_to_excel(assoc_rules, output_file_name, type, execution_counter=execution_counter)
    return assoc_rules


def get_percentage_of_similar_association_rules(assoc_rules_orig, assoc_rules_synth):
    antecedents_synth = list(assoc_rules_synth['antecedents'])
    matched_rules_count = 0

    if len(assoc_rules_orig) > 0:
        for index, row in assoc_rules_orig.iterrows():
            match = 0
            if row['antecedents'] in antecedents_synth:
                matched_rows_synth = assoc_rules_synth.loc[assoc_rules_synth['antecedents'] == row['antecedents']]
                for index_synth, row_synth in matched_rows_synth.iterrows():
                    if row_synth['consequents'] == row['consequents']:
                        match = 1
                matched_rules_count += match

        matched_rules_percentage = matched_rules_count / len(assoc_rules_orig)
        return matched_rules_percentage
    else:
        return 0


def fill_na_cells(df, all_relevant_categorical_columns, all_relevant_continuous_columns):
    imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
    df[all_relevant_categorical_columns] = imputer.fit_transform(df[all_relevant_categorical_columns])

    imputer = SimpleImputer(missing_values=np.NaN, strategy='median')
    df[all_relevant_continuous_columns] = imputer.fit_transform(df[all_relevant_continuous_columns])
    return df


def calculate_accuracy_and_f1_score_for_applicability(result_df, accuracy_orig_col_name, f1_score_orig_col_name, accuracy_synth_col_name, f1_score_synth_col_name, original_df, synthetic_df, categorical_columns, bool_columns, feature_columns, target_column, continuous_columns):
    all_relevant_columns = feature_columns.copy()
    all_relevant_columns.append(target_column)
    all_relevant_categorical_columns = list(set(categorical_columns + bool_columns) & set(all_relevant_columns))
    all_relevant_continuous_columns = list(set(continuous_columns) & set(all_relevant_columns))

    original_df = original_df[all_relevant_columns].copy()
    synthetic_df = synthetic_df[all_relevant_columns].copy()
    mapping, inverted_mapping = get_mapping_for_categorical_fields_to_numerical(original_df, all_relevant_categorical_columns)
    map_categorical_fields_to_numerical_values(original_df, all_relevant_categorical_columns, mapping)
    map_categorical_fields_to_numerical_values(synthetic_df, all_relevant_categorical_columns, mapping)
    original_df = original_df.dropna()
    synthetic_df = synthetic_df.dropna()

    accuracy_orig_result, f1_score_orig_result = generate_decision_tree_and_evaluate(original_df, feature_columns, target_column)
    accuracy_synth_result, f1_score_synth_result = generate_decision_tree_and_evaluate(synthetic_df, feature_columns, target_column)
    result_df['Decision_Tree_{}'.format(accuracy_orig_col_name)] = [accuracy_orig_result]
    result_df['Decision_Tree_{}'.format(f1_score_orig_col_name)] = [f1_score_orig_result]
    result_df['Decision_Tree_{}'.format(accuracy_synth_col_name)] = [accuracy_synth_result]
    result_df['Decision_Tree_{}'.format(f1_score_synth_col_name)] = [f1_score_synth_result]

    accuracy_orig_result, f1_score_orig_result = generate_support_vector_machine_and_evaluate(original_df, feature_columns, target_column)
    accuracy_synth_result, f1_score_synth_result = generate_support_vector_machine_and_evaluate(synthetic_df, feature_columns, target_column)
    result_df['SVM_{}'.format(accuracy_orig_col_name)] = [accuracy_orig_result]
    result_df['SVM_{}'.format(f1_score_orig_col_name)] = [f1_score_orig_result]
    result_df['SVM_{}'.format(accuracy_synth_col_name)] = [accuracy_synth_result]
    result_df['SVM_{}'.format(f1_score_synth_col_name)] = [f1_score_synth_result]

    accuracy_orig_result, f1_score_orig_result = generate_logistic_regression_and_evaluate(original_df, feature_columns, target_column)
    accuracy_synth_result, f1_score_synth_result = generate_logistic_regression_and_evaluate(synthetic_df, feature_columns, target_column)
    result_df['Log_Regression_{}'.format(accuracy_orig_col_name)] = [accuracy_orig_result]
    result_df['Log_Regression_{}'.format(f1_score_orig_col_name)] = [f1_score_orig_result]
    result_df['Log_Regression_{}'.format(accuracy_synth_col_name)] = [accuracy_synth_result]
    result_df['Log_Regression_{}'.format(f1_score_synth_col_name)] = [f1_score_synth_result]

    accuracy_orig_result, f1_score_orig_result = generate_random_forest_and_evaluate(original_df, feature_columns, target_column)
    accuracy_synth_result, f1_score_synth_result = generate_random_forest_and_evaluate(synthetic_df, feature_columns, target_column)
    result_df['Random_Forest_{}'.format(accuracy_orig_col_name)] = [accuracy_orig_result]
    result_df['Random_Forest_{}'.format(f1_score_orig_col_name)] = [f1_score_orig_result]
    result_df['Random_Forest_{}'.format(accuracy_synth_col_name)] = [accuracy_synth_result]
    result_df['Random_Forest_{}'.format(f1_score_synth_col_name)] = [f1_score_synth_result]
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


def get_synthetic_precision_as_a_whole(original_df, synthetic_df, csv_file_name, synthetic_method, datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column):
    df = pd.DataFrame()

    df['CSV-Datei'] = [csv_file_name]
    df['Methode'] = [synthetic_method]

    df.set_index('CSV-Datei', inplace=True)

    # Generate heatmap with the correlation values for continous and numerical values
    # 1.0 the two dimensions do correlate, 0.0 the two dimensions do not correlate
    generate_heatmap(original_df, synthetic_df, csv_file_name, synthetic_method, datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column)

    # Calculate likelihood that the synthetic data belongs to the real data
    # high value: the synthetic data has high possible likelihood of belonging to the real data
    # low value: the synthetic data has the low possible likelihood of belonging to the real data
    # https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/data-likelihood/gmlikelihood
    original_df_copy = original_df.copy()
    synthetic_df_copy = synthetic_df.copy()
    if re.search('.*{}'.format('Faker_Jö_Bonusclub_Output.csv'), csv_file_name):
        original_df_copy.drop('Telefonnummer', inplace=True, axis=1)
        synthetic_df_copy.drop('Telefonnummer', inplace=True, axis=1)
    df['Gaussian_Mixture_Wahrscheinlichkeit'] = [gaussian_mixture_likelihood(original_df_copy, synthetic_df_copy)]
    return df


def generate_heatmap(original_df, synthetic_df, csv_file_name, synthetic_method, datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column):
    fig, ax = plt.subplots(figsize=(15, 15))
    nominal.associations(original_df[continuous_columns + bool_columns + categorical_columns + modified_datetime_column], nominal_columns=categorical_columns + bool_columns + modified_datetime_column, ax=ax, cmap='coolwarm', annot=True, plot=False)
    fig.savefig(output_folder / '{}_original.png'.format(csv_file_name[:-4]), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 15))
    nominal.associations(synthetic_df[continuous_columns + bool_columns + categorical_columns + modified_datetime_column], nominal_columns=categorical_columns + bool_columns + modified_datetime_column, ax=ax, cmap='coolwarm', annot=True, plot=False)
    fig.savefig(output_folder / '{}_{}.png'.format(csv_file_name[:-4], synthetic_method), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 15))
    nominal.associations(original_df[continuous_columns + bool_columns + categorical_columns + modified_datetime_column], nominal_columns=categorical_columns + bool_columns + modified_datetime_column, nom_nom_assoc='theil', ax=ax, cmap='coolwarm', annot=True, plot=False)
    fig.savefig(output_folder / '{}_theil_original.png'.format(csv_file_name[:-4]), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 15))
    nominal.associations(synthetic_df[continuous_columns + bool_columns + categorical_columns + modified_datetime_column], nominal_columns=categorical_columns + bool_columns + modified_datetime_column, nom_nom_assoc='theil', ax=ax, cmap='coolwarm', annot=True, plot=False)
    fig.savefig(output_folder / '{}_{}_theil.png'.format(csv_file_name[:-4], synthetic_method), dpi=300, bbox_inches='tight')
    plt.close()


def statisical_similarity(original_df, synthetic_df, column_name, statistic_method):
    return (sdmetrics.single_column.StatisticSimilarity.compute(
        real_data=original_df[column_name],
        synthetic_data=synthetic_df[column_name],
        statistic=statistic_method
    ))


def category_coverage(original_df, synthetic_df, column_name):
    return (sdmetrics.single_column.CategoryCoverage.compute(
        real_data=original_df[column_name],
        synthetic_data=synthetic_df[column_name]
    ))


def range_coverage(original_df, synthetic_df, column_name):
    return (sdmetrics.single_column.RangeCoverage.compute(
        real_data=original_df[column_name],
        synthetic_data=synthetic_df[column_name]
    ))


def boundary_adherence(original_df, synthetic_df, column_name):
    return (sdmetrics.single_column.BoundaryAdherence.compute(
        real_data=original_df[column_name],
        synthetic_data=synthetic_df[column_name]
    ))


def category_frequency(original_df, synthetic_df, column_name):
    return (sdmetrics.single_column.TVComplement.compute(
        real_data=original_df[column_name],
        synthetic_data=synthetic_df[column_name]
    ))


def missing_value_proportion(original_df, synthetic_df, column_name):
    return (sdmetrics.single_column.MissingValueSimilarity.compute(
        real_data=original_df[column_name],
        synthetic_data=synthetic_df[column_name]
    ))


def gaussian_mixture_likelihood(original_df, synthetic_df):
    return (GMLogLikelihood.compute(
        real_data=original_df,
        synthetic_data=synthetic_df
    ))


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


def write_multiple_df_to_excel(dataframes, file_name, section_name, execution_counter=1):
    section_name = section_name[:30]
    output_file = output_folder / file_name
    if execution_counter != 0 and check_if_file_exists(output_file):
        mode = 'a'
    else:
        mode = 'w'
    with pd.ExcelWriter(output_file, engine='xlsxwriter', mode=mode) as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet(section_name)
        writer.sheets[section_name] = worksheet

        column = 0
        row = 0

        for df in dataframes:
            df.to_excel(writer, sheet_name=section_name, startrow=row, startcol=column)
            row += df.shape[0] + 2
        print('\n\t\t[-] Results saved to "{}".'.format(output_file))


def add_modified_date_column(df, csv_file_name):
    csv_file_name = str(csv_file_name)
    if re.search('.*{}'.format('Case_Information.csv'), csv_file_name):
        df['date_announced_month'] = [calendar.month_name[d.month] if d is not pd.NaT else pd.NaT for d in df['date_announced']]
        df['date_recovered_month'] = [calendar.month_name[d.month] if d is not pd.NaT else pd.NaT for d in df['date_recovered']]
        df['date_of_death_month'] = [calendar.month_name[d.month] if d is not pd.NaT else pd.NaT for d in df['date_of_death']]
        df['date_announced_as_removed_month'] = [calendar.month_name[d.month] if d is not pd.NaT else pd.NaT for d in df['date_announced_as_removed']]
        df['date_of_onset_of_symptoms_month'] = [calendar.month_name[d.month] if d is not pd.NaT else pd.NaT for d in df['date_of_onset_of_symptoms']]
    elif re.search('.*{}'.format('Faker_Jö_Bonusclub_Output.csv'), csv_file_name):
        df['Kaufdatum_day'] = [d.day_name() if d is not pd.NaT else pd.NaT for d in df['Kaufdatum']]
        df['Geburtsdatum_year'] = [d.year if d is not pd.NaT else pd.NaT for d in df['Geburtsdatum']]
    elif re.search('.*{}'.format('Online Retail.csv'), csv_file_name):
        df['InvoiceDate_month'] = [calendar.month_name[d.month] if d is not pd.NaT else pd.NaT for d in df['InvoiceDate']]
        df['InvoiceDate_date'] = [d.strftime('%d.%m.%Y') if d is not pd.NaT else pd.NaT for d in df['InvoiceDate']]
    elif re.search('.*{}'.format('Faker_Elektronischer_Impfpass_Output.csv'), csv_file_name):
        df['Geburtsdatum_year'] = [d.year if d is not pd.NaT else pd.NaT for d in df['Geburtsdatum']]
        df['Tetanus und Diphterie Impfdatum_month'] = [calendar.month_name[d.month] if d is not pd.NaT else pd.NaT for d in df['Tetanus und Diphterie Impfdatum']]
        df['Tetanus und Diphterie Ablaufdatum des Impfstoffs_year'] = [d.year if d is not pd.NaT else pd.NaT for d in df['Tetanus und Diphterie Ablaufdatum des Impfstoffs']]
        df['Polio Impfdatum_month'] = [calendar.month_name[d.month] if d is not pd.NaT else pd.NaT for d in df['Polio Impfdatum']]
        df['Polio Ablaufdatum des Impfstoffs_year'] = [d.year if d is not pd.NaT else pd.NaT for d in df['Polio Ablaufdatum des Impfstoffs']]
        df['FSME Impfdatum_month'] = [calendar.month_name[d.month] if d is not pd.NaT else pd.NaT for d in df['FSME Impfdatum']]
        df['FSME Ablaufdatum des Impfstoffs_year'] = [d.year if d is not pd.NaT else pd.NaT for d in df['FSME Ablaufdatum des Impfstoffs']]
        df['Covid Impfdatum_month'] = [calendar.month_name[d.month] if d is not pd.NaT else pd.NaT for d in df['Covid Impfdatum']]
        df['Covid Ablaufdatum des Impfstoffs_year'] = [d.year if d is not pd.NaT else pd.NaT for d in df['Covid Ablaufdatum des Impfstoffs']]
    else:
        print('\n\t[!] Got unsupported csv-file "{}"!'.format(csv_file_name))
        sys.exit(1)
    return df


def convert_specific_columns_to_boolean(input_file, df):
    if input_file == 'Case_Information.csv':
        df['home_quarantined'] = df['home_quarantined'].map({'Yes': True, 'No': False})
        df['pregnant'] = df['pregnant'].map({'Yes': True, 'No': False})
        return df
    elif input_file == 'Faker_Elektronischer_Impfpass_Output.csv':
        df['Tetanus und Diphterie Impfung'] = df['Tetanus und Diphterie Impfung'].map({1: True, 0: False})
        df['Polio Impfung'] = df['Polio Impfung'].map({1: True, 0: False})
        df['FSME Impfung'] = df['FSME Impfung'].map({1: True, 0: False})
        df['Covid Impfung'] = df['Covid Impfung'].map({1: True, 0: False})
        df.head(10)
        return df
    return df


def get_column_types(csv_file_name):
    if csv_file_name == 'Case_Information.csv':
        datetime_columns = ['date_announced', 'date_recovered', 'date_of_death', 'date_announced_as_removed', 'date_of_onset_of_symptoms']
        continuous_columns = ['age']
        categorical_columns = ['sex', 'status', 'province', 'muni_city', 'health_status', 'region']
        bool_columns = ['home_quarantined', 'pregnant']
        ignore_columns = ['case_id']
        modified_datetime_column = ['date_announced_month', 'date_recovered_month', 'date_of_death_month', 'date_announced_as_removed_month', 'date_of_onset_of_symptoms_month']
    elif csv_file_name == 'Faker_Jö_Bonusclub_Output.csv':
        datetime_columns = ['Kaufdatum', 'Geburtsdatum']
        continuous_columns = ['Preis in € je Stk', 'Anzahl', 'Gesamtpreis', 'Gesammelte Ös']
        categorical_columns = ['Produktname', 'Kategorie', 'Verbrauchausgaben-Kategorie', 'Bezahlart', 'Geschlecht', 'Postleitzahl', 'Ort']
        bool_columns = []
        ignore_columns = ['Mitgliedsnummer', 'Vorname', 'Nachname', 'Straße', 'Telefonnummer', 'E-Mailadresse']
        modified_datetime_column = ['Kaufdatum_day', 'Geburtsdatum_year']
    elif csv_file_name == 'Online Retail.csv':
        datetime_columns = ['InvoiceDate']
        continuous_columns = ['Quantity', 'UnitPrice', 'CustomerID']
        categorical_columns = ['StockCode', 'Description', 'Country']
        bool_columns = []
        ignore_columns = ['InvoiceNo']
        modified_datetime_column = ['InvoiceDate_month']
    elif csv_file_name == 'Faker_Elektronischer_Impfpass_Output.csv':
        datetime_columns = ['Geburtsdatum', 'Tetanus und Diphterie Impfdatum', 'Tetanus und Diphterie Ablaufdatum des Impfstoffs', 'Polio Impfdatum', 'Polio Ablaufdatum des Impfstoffs', 'FSME Impfdatum', 'FSME Ablaufdatum des Impfstoffs', 'Covid Impfdatum', 'Covid Ablaufdatum des Impfstoffs']
        continuous_columns = []
        categorical_columns = ['Geschlecht', 'Bundesland', 'Postleitzahl', 'Ort', 'Tetanus und Diphterie Impfstoff', 'Tetanus und Diphterie Impfstoffhersteller', 'Polio Impfstoff', 'Polio Impfstoffhersteller', 'FSME Impfstoff', 'FSME Impfstoffhersteller', 'Covid Impfstoff', 'Covid Impfstoffhersteller']
        bool_columns = ['Tetanus und Diphterie Impfung', 'Polio Impfung', 'FSME Impfung', 'Covid Impfung']
        ignore_columns = ['Vorname', 'Nachname', 'Sozialversicherungsnummer', 'Straße']
        modified_datetime_column = ['Geburtsdatum_year', 'Tetanus und Diphterie Impfdatum_month', 'Tetanus und Diphterie Ablaufdatum des Impfstoffs_year', 'Polio Impfdatum_month', 'Polio Ablaufdatum des Impfstoffs_year', 'FSME Impfdatum_month', 'FSME Ablaufdatum des Impfstoffs_year', 'Covid Impfdatum_month', 'Covid Ablaufdatum des Impfstoffs_year']
    else:
        print('\n\t[!] Got unsupported csv-file "{}"!'.format(csv_file_name))
        sys.exit(1)
    return datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column


def main():
    stdout_backup = sys.stdout
    sys.stdout = UnbufferedLogging(sys.stdout)

    check_if_required_dirs_exist()
    csv_files = get_list_of_input_csv()

    print('\n\t[1] Calculating structural similarity...')
    counter = 0
    for csv_file in csv_files:
        datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column = get_column_types(csv_file)
        all_structural_results_df = pd.DataFrame()

        for synthetic_csv_file in original_synthetic_file_mapping[csv_file]:
            original_df = read_csv_file(original_input_folder / csv_file)
            synthetic_df = read_csv_file(synthetic_input_folder / synthetic_csv_file)
            structural_results_df = get_structural_similarity(original_df, synthetic_df, csv_file, synthetic_csv_file.split('_')[0], datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column)
            all_structural_results_df = pd.concat([all_structural_results_df, structural_results_df])

        write_df_to_excel(all_structural_results_df, 'Validation_result_1_structure.xlsx', csv_file[:-4], counter)
        counter += 1

    print('\n\t[2] Calculating bivariate accuracy...')
    for csv_file in csv_files:
        datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column = get_column_types(csv_file)
        for synthetic_csv_file in original_synthetic_file_mapping[csv_file]:
            original_df = read_csv_file(original_input_folder / csv_file)
            synthetic_df = read_csv_file(synthetic_input_folder / synthetic_csv_file)
            get_synthetic_bivariate_accuracy(original_df, synthetic_df, csv_file, synthetic_csv_file.split('_')[0], datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column)

    print('\n\t[3] Calculating precision as a whole...')
    counter = 0
    for csv_file in csv_files:
        datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column = get_column_types(csv_file)
        all_precision_results_df = pd.DataFrame()
        for synthetic_csv_file in original_synthetic_file_mapping[csv_file]:
            original_df = read_csv_file(original_input_folder / csv_file)
            synthetic_df = read_csv_file(synthetic_input_folder / synthetic_csv_file)
            precision_results_df = get_synthetic_precision_as_a_whole(original_df, synthetic_df, csv_file, synthetic_csv_file.split('_')[0], datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column)
            all_precision_results_df = pd.concat([all_precision_results_df, precision_results_df])

        write_df_to_excel(all_precision_results_df, 'Validation_result_3_precision.xlsx', csv_file[:-4], counter)
        counter += 1

    print('\n\t[4] Checking applicability...')
    counter = 0
    for csv_file in csv_files:
        datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column = get_column_types(csv_file)
        all_applicability_results_df = pd.DataFrame()

        for synthetic_csv_file in original_synthetic_file_mapping[csv_file]:
            original_df = read_csv_file(original_input_folder / csv_file)
            synthetic_df = read_csv_file(synthetic_input_folder / synthetic_csv_file)
            applicability_results_df = check_applicability(original_df, synthetic_df, csv_file, synthetic_csv_file.split('_')[0], datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column)
            all_applicability_results_df = pd.concat([all_applicability_results_df, applicability_results_df])

        write_df_to_excel(all_applicability_results_df, 'Validation_result_4_applicability.xlsx', csv_file[:-4], counter)
        counter += 1

    print('\n\t[5] Checking privacy...')
    for csv_file in csv_files:
        datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column = get_column_types(
            csv_file)
        for synthetic_csv_file in original_synthetic_file_mapping[csv_file]:
            original_df = read_csv_file(original_input_folder / csv_file)
            synthetic_df = read_csv_file(synthetic_input_folder / synthetic_csv_file)
            check_synthetic_privacy(original_df, synthetic_df, csv_file, synthetic_csv_file.split('_')[0], datetime_columns, continuous_columns, categorical_columns, bool_columns, ignore_columns, modified_datetime_column)

    sys.stdout = stdout_backup
    log_file_handler.close()


if __name__ == "__main__":
    main()
