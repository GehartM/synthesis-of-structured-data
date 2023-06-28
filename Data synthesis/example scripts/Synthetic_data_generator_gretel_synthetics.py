from sdv.constraints import FixedCombinations
from sdv.constraints.base import Constraint
from gretel_synthetics.actgan import ACTGAN
import faker
import pandas as pd
from collections import namedtuple
import os
import sys
import csv
import gc
import torch
from pathlib import Path
import datetime
import argparse


faker.proxy.DEFAULT_LOCALE = 'de_AT'
input_folder = Path('Input')
output_folder = Path('Output')
epochs = 100
batch_size = 4000
command_line_args = False

gc.collect()
torch.cuda.empty_cache()
log_file_handler = open('synthetic_data_generator_{}.log'.format(str(datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))), 'w')


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


def print_list_of_available_synthesis_methods():
    print('\n\t[-] The following synthesis methods are supported:')
    for i, synthetic_method in enumerate(supported_synthetic_methods):
        print('\t\t[{}] {}'.format(i, synthetic_method.name))


def get_desired_synthesis_methods(args):
    desired_synthesis_methods = set()
    while len(desired_synthesis_methods) <= 0:
        if args.methods is None:
            choice = input('\tChoose which synthesis methods should be applied (e.g. 1, 2, 4-6): ')
            choice = choice.split(',')
        else:
            choice = args.methods
            choice = choice.split(',')

        for number in choice:
            number = ''.join(number.split())
            if number.isdigit():
                desired_synthesis_methods.add(int(number))
            else:
                numbers = number.split('-')
                if len(numbers) == 2 and numbers[0].isdigit() and numbers[1].isdigit():
                    for i in range(int(numbers[0]), int(numbers[1])+1):
                        desired_synthesis_methods.add(i)
                else:
                    print('\t[!] Got invalid input: "{}"'.format(number))
                    desired_synthesis_methods = set()
                    if args.methods is not None:
                        sys.exit(1)
                    continue

        for method in desired_synthesis_methods:
            if not 0 <= method < len(supported_synthetic_methods):
                print('\t[!] Number "{}" is out of range!'.format(method))
                desired_synthesis_methods = set()
                if args.methods is not None:
                    sys.exit(1)
    return desired_synthesis_methods


def read_csv_file(input_file):
    input_file = input_folder / input_file
    delimiter = get_delimiter_of_csv(input_file)
    if delimiter is None:
        print('\n\t[!] The delimiter for "{}" could not be automatically determined!'.format(input_file))
        if command_line_args is False:
            while delimiter is None:
                delimiter = input('\tEnter the delimiter for this file: ')
                if len(delimiter) != 1:
                    print('\t[!] The delimiter can only be one character!')
                    delimiter = None
        else:
            print('\n\t[!] Skipping "{}"!'.format(input_file))
            return None
    try:
        return pd.read_csv(input_file, sep=delimiter, decimal=',')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(input_file, sep=delimiter, decimal=',', encoding='utf-8-sig')
        except UnicodeDecodeError:
            return pd.read_csv(input_file, sep=delimiter, decimal=',', encoding='ISO-8859-1')


def synthesise_with_cgan():
    pass


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


def convert_specific_columns_to_object(input_file, df):
    if input_file == 'Faker_Jö_Bonusclub_Output.csv':
        df['Telefonnummer'] = df['Telefonnummer'].astype(object)
        return df
    return df


def synthesise_with_actgan(input_file):
    print('\n\t[-] Starting synthesis of "{}" with actGAN...'.format(input_file))
    df = read_csv_file(input_file)

    if df is not None:
        df = convert_specific_columns_to_boolean(input_file, df)
        df = convert_specific_columns_to_object(input_file, df)
        field_types = None
        anonymize_fields = None

        if input_file == 'Case_Information.csv':
            field_types = {
                'case_id': {'type': 'id', 'subtype': 'string', 'regex': 'C[0-9]{6}'},
                'age': {'type': 'numerical', 'subtype': 'integer'},
                'sex': {'type': 'categorical'},
                'date_announced': {'type': 'datetime', 'format': '%d.%m.%Y'},
                'date_recovered': {'type': 'datetime', 'format': '%d.%m.%Y'},
                'date_of_death': {'type': 'datetime', 'format': '%d.%m.%Y'},
                'status': {'type': 'categorical'},
                'date_announced_as_removed': {'type': 'datetime', 'format': '%d.%m.%Y'},
                'province': {'type': 'categorical'},
                'muni_city': {'type': 'categorical'},
                'health_status': {'type': 'categorical'},
                'home_quarantined': {'type': 'boolean'},
                'date_of_onset_of_symptoms': {'type': 'datetime', 'format': '%d.%m.%Y'},
                'pregnant': {'type': 'boolean'},
                'region': {'type': 'categorical'}
            }
        elif input_file == 'Online Retail.csv':
            field_types = {
                'InvoiceNo': {'type': 'categorical'},
                'StockCode': {'type': 'categorical'},
                'Description': {'type': 'categorical'},
                'Quantity': {'type': 'numerical', 'subtype': 'integer'},
                #'InvoiceDate': {'type': 'datetime', 'format': '%d.%m.%Y %H:%M'},
                'UnitPrice': {'type': 'categorical'},
                'CustomerID': {'type': 'numerical', 'subtype': 'integer'},
                'Country': {'type': 'categorical'}
            }
            print(df['InvoiceDate'])
        elif input_file == 'Faker_Elektronischer_Impfpass_Output.csv':
            field_types = {
                'Vorname': {'type': 'categorical', 'pii': 'true', 'pii_category': 'first_name',
                            'pii_locales': ['de_AT']},
                'Nachname': {'type': 'categorical', 'pii': 'true', 'pii_category': 'last_name',
                             'pii_locales': ['de_AT']},
                'Geschlecht': {'type': 'categorical'},
                'Geburtsdatum': {'type': 'datetime', 'format': '%Y-%m-%d'},
                'Sozialversicherungsnummer': {'type': 'numerical', 'subtype': 'integer'},
                'Bundesland': {'type': 'categorical'},
                'Postleitzahl': {'type': 'categorical'},
                'Ort': {'type': 'categorical'},
                'Straße': {'type': 'categorical', 'pii': 'true', 'pii_category': 'street_address',
                           'pii_locales': ['de_AT']},
                'Tetanus und Diphterie Impfung': {'type': 'boolean'},
                'Tetanus und Diphterie Impfstoff': {'type': 'categorical'},
                'Tetanus und Diphterie Impfstoffhersteller': {'type': 'categorical'},
                'Tetanus und Diphterie Impfdatum': {'type': 'datetime', 'format': '%Y-%m-%d'},
                'Tetanus und Diphterie Ablaufdatum des Impfstoffs': {'type': 'datetime', 'format': '%Y-%m-%d'},
                'Polio Impfung': {'type': 'boolean'},
                'Polio Impfstoff': {'type': 'categorical'},
                'Polio Impfstoffhersteller': {'type': 'categorical'},
                'Polio Impfdatum': {'type': 'datetime', 'format': '%Y-%m-%d'},
                'Polio Ablaufdatum des Impfstoffs': {'type': 'datetime', 'format': '%Y-%m-%d'},
                'FSME Impfung': {'type': 'boolean'},
                'FSME Impfstoff': {'type': 'categorical'},
                'FSME Impfstoffhersteller': {'type': 'categorical'},
                'FSME Impfdatum': {'type': 'datetime', 'format': '%Y-%m-%d'},
                'FSME Ablaufdatum des Impfstoffs': {'type': 'datetime', 'format': '%Y-%m-%d'},
                'Covid Impfung': {'type': 'boolean'},
                'Covid Impfstoff': {'type': 'categorical'},
                'Covid Impfstoffhersteller': {'type': 'categorical'},
                'Covid Impfdatum': {'type': 'datetime', 'format': '%Y-%m-%d'},
                'Covid Ablaufdatum des Impfstoffs': {'type': 'datetime', 'format': '%Y-%m-%d'}
            }
        elif input_file == 'Faker_Jö_Bonusclub_Output.csv':
            field_types = {
                'Produktname': {'type': 'categorical'},
                'Kategorie': {'type': 'categorical'},
                'Verbrauchausgaben-Kategorie': {'type': 'categorical'},
                'Preis in € je Stk': {'type': 'categorical'},
                'Anzahl': {'type': 'numerical', 'subtype': 'integer'},
                'Gesamtpreis': {'type': 'numerical', 'subtype': 'float'},
                'Gesammelte Ös': {'type': 'numerical', 'subtype': 'integer'},
                'Kaufdatum': {'type': 'datetime', 'format': '%d.%m.%Y %H:%M'},
                'Bezahlart': {'type': 'categorical'},
                'Mitgliedsnummer': {'type': 'numerical', 'subtype': 'integer'},
                'Vorname': {'type': 'categorical', 'pii': 'true', 'pii_category': 'first_name',
                            'pii_locales': ['de_AT']},
                'Nachname': {'type': 'categorical', 'pii': 'true', 'pii_category': 'last_name',
                             'pii_locales': ['de_AT']},
                'Geschlecht': {'type': 'categorical'},
                'Geburtsdatum': {'type': 'datetime', 'format': '%d.%m.%Y'},
                'Postleitzahl': {'type': 'categorical'},
                'Ort': {'type': 'categorical'},
                'Straße': {'type': 'categorical', 'pii': 'true', 'pii_category': 'street_address',
                           'pii_locales': ['de_AT']},
                'Telefonnummer': {'type': 'categorical', 'pii': 'true', 'pii_category': 'phone_number',
                                  'pii_locales': ['de_AT']},
                'E-Mailadresse': {'type': 'categorical', 'pii': 'true', 'pii_category': 'free_email',
                                  'pii_locales': ['de_AT']}
            }
            anonymize_fields = {
                'Vorname': 'country',
                'Nachname': 'last_name',
                'Straße': 'street_address',
                'Telefonnummer': 'phone_number',
                'E-Mailadresse': 'free_email'
            }

        added_constraints = sdv_add_constraints_for_specific_csv_file(input_file)
        model = ACTGAN(
            field_types=field_types,
            constraints=added_constraints,
            verbose=True,
            binary_encoder_cutoff=10,
            auto_transform_datetimes=True,
            epochs=epochs,
            anonymize_fields=anonymize_fields
        )
        print('\t\t[.] Fitting the model...')
        model.fit(df)

        print('\t\t[.] Saving the model...')
        model.save('{}.pkl'.format(output_folder / '{}_{}'.format('actGAN', input_file[:-4])))
        sample_count = len(df.index)
        print('\t\t[.] Generating {} samples from fitted model...'.format(sample_count))
        synthetic_df = model.sample(num_rows=sample_count)
        output_file = save_synthetic_data_as_csv(synthetic_df, input_file, 'actGAN')
        print('\t\t[.] {} samples saved to "{}".'.format(sample_count, output_file))


def sdv_add_constraints_for_specific_csv_file(input_file):
    added_constraints = list()
    if input_file == 'Case_Information.csv':
        added_constraints.append(FixedCombinations(['status', 'health_status']))
        added_constraints.append(FixedCombinations(['province', 'muni_city', 'region']))
        added_constraints.append(Custom_Constraint_Case_Information_Recovery_Death_Date([]))
    elif input_file == 'Online Retail.csv':
        added_constraints.append(FixedCombinations(['StockCode', 'Description', 'UnitPrice']))
    elif input_file == 'Faker_Elektronischer_Impfpass_Output.csv':
        added_constraints.append(FixedCombinations(['Bundesland', 'Postleitzahl', 'Ort']))
        added_constraints.append(FixedCombinations(['Tetanus und Diphterie Impfstoff', 'Tetanus und Diphterie Impfstoffhersteller']))
        added_constraints.append(FixedCombinations(['Polio Impfstoff', 'Polio Impfstoffhersteller']))
        added_constraints.append(FixedCombinations(['FSME Impfstoff', 'FSME Impfstoffhersteller']))
        added_constraints.append(FixedCombinations(['Covid Impfstoff', 'Covid Impfstoffhersteller']))
        added_constraints.append(Custom_Constraint_Elektronischer_Impfpass(['Tetanus und Diphterie Impfung', 'Tetanus und Diphterie Impfstoff', 'Tetanus und Diphterie Impfstoffhersteller',
                                 'Tetanus und Diphterie Impfdatum', 'Tetanus und Diphterie Ablaufdatum des Impfstoffs']))
        added_constraints.append(Custom_Constraint_Elektronischer_Impfpass(['Polio Impfung', 'Polio Impfstoff', 'Polio Impfstoffhersteller', 'Polio Impfdatum', 'Polio Ablaufdatum des Impfstoffs']))
        added_constraints.append(Custom_Constraint_Elektronischer_Impfpass(['FSME Impfung', 'FSME Impfstoff', 'FSME Impfstoffhersteller', 'FSME Impfdatum', 'FSME Ablaufdatum des Impfstoffs']))
        added_constraints.append(Custom_Constraint_Elektronischer_Impfpass(['Covid Impfung', 'Covid Impfstoff', 'Covid Impfstoffhersteller', 'Covid Impfdatum', 'Covid Ablaufdatum des Impfstoffs']))
    elif input_file == 'Faker_Jö_Bonusclub_Output.csv':
        added_constraints.append(FixedCombinations(['Produktname', 'Kategorie', 'Verbrauchausgaben-Kategorie', 'Preis in € je Stk']))
        added_constraints.append(FixedCombinations(['Postleitzahl', 'Ort']))
    return added_constraints


def save_synthetic_data_as_csv(synthetic_df, output_filename, synthetic_method):
    output_filename = output_folder / '{}_{}'.format(synthetic_method, output_filename)
    synthetic_df.to_csv(output_filename, index=False, encoding='utf-8-sig', sep=';', decimal=',')
    return output_filename


Synthetic_method = namedtuple('Synthetic_method', 'name function')
supported_synthetic_methods = [
    Synthetic_method('actGAN', synthesise_with_actgan)
]


def get_arguments():
    global command_line_args
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', '--methods', type=str, help='Synthetic methods to be used')
    args = argParser.parse_args()
    command_line_args = True
    return args


class Custom_Constraint_Case_Information_Recovery_Death_Date(Constraint):
    def __init__(self, column_name):
        self._column_name = column_name

    def is_valid(self, data):
        return(((data['status'] == 'Died') & (data['date_recovered'].isnull())) | (
                    (data['status'] != 'Died') & (data['date_of_death'].isnull())))


class Custom_Constraint_Elektronischer_Impfpass(Constraint):
    def __init__(self, column_name):
        self._column_name = column_name

    def is_valid(self, data):
        vaccination = self._column_name[0]
        vaccine = self._column_name[1]
        manufacturer = self._column_name[2]
        vaccination_date = self._column_name[3]
        expiry_date = self._column_name[4]

        return (
            (((~data[vaccination]) & (data[vaccine].isnull()) &
              (data[manufacturer].isnull()) & (data[vaccination_date].isnull()) &
              (data[expiry_date].isnull())) |
             ((data[vaccination]) & (~ data[vaccine].isnull()) &
              (~ data[manufacturer].isnull()) & (~ data[vaccination_date].isnull()) &
              (~ data[expiry_date].isnull()) & (data[vaccination_date] < data[expiry_date])))
        )


def main():
    stdout_backup = sys.stdout
    sys.stdout = UnbufferedLogging(sys.stdout)

    args = get_arguments()
    check_if_required_dirs_exist()
    csv_files = get_list_of_input_csv()
    print_list_of_available_synthesis_methods()
    synthetic_methods = get_desired_synthesis_methods(args)
    for csv_file in csv_files:
        for method in synthetic_methods:
            supported_synthetic_methods[method].function(csv_file)

    sys.stdout = stdout_backup
    log_file_handler.close()


if __name__ == "__main__":
    main()
