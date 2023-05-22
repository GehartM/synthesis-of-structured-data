from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
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
epochs = 100 # 100
batch_size = 4000 # 4000
# Case Information -> batch size 4000
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


def synthesise_with_ctgan(input_file):
    print('\n\t[-] Starting synthesis of "{}" with CTGAN...'.format(input_file))
    df = read_csv_file(input_file)

    if df is not None:
        df = convert_specific_columns_to_boolean(input_file, df)
        df = convert_specific_columns_to_object(input_file, df)

        metadata_file = input_folder / '{}_Metadata.json'.format(input_file[:-4])
        if check_if_file_exists(metadata_file):
            metadata = SingleTableMetadata.load_from_json(filepath=metadata_file)
            model = CTGANSynthesizer(metadata, batch_size=batch_size, epochs=epochs, verbose=True)
            if input_file == 'Case_Information.csv':
                model.load_custom_constraint_classes('Custom_constraints_Case_Information.py',
                                                     class_names=['Custom_Constraint_Case_Information_Recovery_Death_Date'])
            elif input_file == 'Faker_Elektronischer_Impfpass_Output.csv':
                model.load_custom_constraint_classes('Custom_constraints_Elektronischer_Impfpass.py',
                                                     class_names=[
                                                         'Custom_Constraint_Elektronischer_Impfpass'])
            added_constraints = sdv_add_constraints_for_specific_csv_file(input_file)
            model.add_constraints(added_constraints)
            print('\t\t[.] Fitting the model...')
            model.fit(df)

            print('\t\t[.] Saving the model...')
            model.save(filepath='{}.pkl'.format(output_folder / '{}_{}'.format('CTGAN', input_file[:-4])))
            sample_count = len(df.index)
            print('\t\t[.] Generating {} samples from fitted model...'.format(sample_count))
            synthetic_df = model.sample(num_rows=sample_count)
            output_file = save_synthetic_data_as_csv(synthetic_df, input_file, 'CTGAN')
            print('\t\t[.] {} samples saved to "{}".'.format(sample_count, output_file))
        else:
            print('\t\t[!] The required metadata file "{}" for "{}" does not exist!'.format(metadata_file, input_file))
            print('\n\t[!] Skipping "{}"!'.format(input_file))


def synthesise_with_tvae(input_file):
    print('\n\t[-] Starting synthesis of "{}" with TVAE...'.format(input_file))
    df = read_csv_file(input_file)
    if df is not None:
        df = convert_specific_columns_to_boolean(input_file, df)
        df = convert_specific_columns_to_object(input_file, df)

        metadata_file = input_folder / '{}_Metadata.json'.format(input_file[:-4])
        if check_if_file_exists(metadata_file):
            metadata = SingleTableMetadata.load_from_json(filepath=metadata_file)
            model = TVAESynthesizer(metadata, batch_size=batch_size, epochs=epochs)
            if input_file == 'Case_Information.csv':
                model.load_custom_constraint_classes('Custom_constraints_Case_Information.py',
                                                     class_names=['Custom_Constraint_Case_Information_Recovery_Death_Date'])
            elif input_file == 'Faker_Elektronischer_Impfpass_Output.csv':
                model.load_custom_constraint_classes('Custom_constraints_Elektronischer_Impfpass.py',
                                                     class_names=[
                                                         'Custom_Constraint_Elektronischer_Impfpass'])
            added_constraints = sdv_add_constraints_for_specific_csv_file(input_file)
            model.add_constraints(added_constraints)
            print('\t\t[.] Fitting the model...')
            model.fit(df)

            print('\t\t[.] Saving the model...')
            model.save(filepath='{}.pkl'.format(output_folder / '{}_{}'.format('TVAE', input_file[:-4])))
            sample_count = len(df.index)
            print('\t\t[.] Generating {} samples from fitted model...'.format(sample_count))
            synthetic_df = model.sample(num_rows=sample_count)
            output_file = save_synthetic_data_as_csv(synthetic_df, input_file, 'TVAE')
            print('\t\t[.] {} samples saved to "{}".'.format(sample_count, output_file))
        else:
            print('\t\t[!] The required metadata file "{}" for "{}" does not exist!'.format(metadata_file, input_file))
            print('\n\t[!] Skipping "{}"!'.format(input_file))


def sdv_add_constraints_for_specific_csv_file(input_file):
    added_constraints = list()
    if input_file == 'Case_Information.csv':
        added_constraints.append({
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {'column_names': ['status', 'health_status']}
        })
        added_constraints.append({
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {'column_names': ['province', 'muni_city', 'region']}
        })
        added_constraints.append({
            'constraint_class': 'Custom_Constraint_Case_Information_Recovery_Death_Date',
            'constraint_parameters': {
                'column_names': []
            }
        })
    elif input_file == 'Online Retail.csv':
        added_constraints.append({
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {'column_names': ['StockCode', 'Description', 'UnitPrice']}
        })
    elif input_file == 'Faker_Elektronischer_Impfpass_Output.csv':
        added_constraints.append({
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {'column_names': ['Bundesland', 'Postleitzahl', 'Ort']}
        })
        added_constraints.append({
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {'column_names': ['Tetanus und Diphterie Impfstoff', 'Tetanus und Diphterie Impfstoffhersteller']}
        })
        added_constraints.append({
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {
                'column_names': ['Polio Impfstoff', 'Polio Impfstoffhersteller']}
        })
        added_constraints.append({
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {
                'column_names': ['FSME Impfstoff', 'FSME Impfstoffhersteller']}
        })
        added_constraints.append({
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {
                'column_names': ['Covid Impfstoff', 'Covid Impfstoffhersteller']}
        })
        added_constraints.append({
            'constraint_class': 'Custom_Constraint_Elektronischer_Impfpass',
            'constraint_parameters': {
                'column_names': ['Tetanus und Diphterie Impfung', 'Tetanus und Diphterie Impfstoff', 'Tetanus und Diphterie Impfstoffhersteller',
                                 'Tetanus und Diphterie Impfdatum', 'Tetanus und Diphterie Ablaufdatum des Impfstoffs']
            }
        })
        added_constraints.append({
            'constraint_class': 'Custom_Constraint_Elektronischer_Impfpass',
            'constraint_parameters': {
                'column_names': ['Polio Impfung', 'Polio Impfstoff', 'Polio Impfstoffhersteller', 'Polio Impfdatum', 'Polio Ablaufdatum des Impfstoffs']
            }
        })
        added_constraints.append({
            'constraint_class': 'Custom_Constraint_Elektronischer_Impfpass',
            'constraint_parameters': {
                'column_names': ['FSME Impfung', 'FSME Impfstoff', 'FSME Impfstoffhersteller', 'FSME Impfdatum', 'FSME Ablaufdatum des Impfstoffs']
            }
        })
        added_constraints.append({
            'constraint_class': 'Custom_Constraint_Elektronischer_Impfpass',
            'constraint_parameters': {
                'column_names': ['Covid Impfung', 'Covid Impfstoff', 'Covid Impfstoffhersteller', 'Covid Impfdatum', 'Covid Ablaufdatum des Impfstoffs']
            }
        })
    elif input_file == 'Faker_Jö_Bonusclub_Output.csv':
        added_constraints.append({
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {'column_names': ['Produktname', 'Kategorie', 'Verbrauchausgaben-Kategorie', 'Preis in € je Stk']}
        })
        added_constraints.append({
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {
                'column_names': ['Postleitzahl', 'Ort']}
        })

    return added_constraints


def save_synthetic_data_as_csv(synthetic_df, output_filename, synthetic_method):
    output_filename = output_folder / '{}_{}'.format(synthetic_method, output_filename)
    synthetic_df.to_csv(output_filename, index=False, encoding='utf-8-sig', sep=';', decimal=',')
    return output_filename


Synthetic_method = namedtuple('Synthetic_method', 'name function')
supported_synthetic_methods = [
    Synthetic_method('CTGAN', synthesise_with_ctgan),
    Synthetic_method('TVAE', synthesise_with_tvae)
]


def get_arguments():
    global command_line_args
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', '--methods', type=str, help='Synthetic methods to be used')
    args = argParser.parse_args()
    command_line_args = True
    return args


def main():
    stdout_backup = sys.stdout
    sys.stdout = UnbufferedLogging(sys.stdout)

    args = get_arguments()
    check_if_required_dirs_exist()
    csv_files = get_list_of_input_csv()
    print_list_of_available_synthesis_methods()
    synthetic_methods = get_desired_synthesis_methods(args)
    #load_synthesise_with_tvae()
    #load_synthesise_with_ctgan()

    for csv_file in csv_files:
        for method in synthetic_methods:
            supported_synthetic_methods[method].function(csv_file)

    sys.stdout = stdout_backup
    log_file_handler.close()


"""
def load_synthesise_with_ctgan():
    input_file = 'Case_Information.csv'
    model = CTGANSynthesizer.load(filepath='CTGAN_Case_Information.pkl')
    print('\t\t[.] Generating {} samples from fitted model...'.format(sample_count))
    synthetic_df = model.sample(num_rows=sample_count)
    output_file = save_synthetic_data_as_csv(synthetic_df, input_file, 'CTGAN')
    print('\t\t[.] {} samples saved to "{}".'.format(sample_count, output_file))


def load_synthesise_with_tvae():
    input_file = 'Case_Information.csv'
    model = TVAESynthesizer.load(filepath='TVAE_Case_Information.pkl')
    print('\t\t[.] Generating {} samples from fitted model...'.format(sample_count))
    synthetic_df = model.sample(num_rows=sample_count)
    output_file = save_synthetic_data_as_csv(synthetic_df, input_file, 'TVAE')
    print('\t\t[.] {} samples saved to "{}".'.format(sample_count, output_file))
"""


if __name__ == "__main__" :
    main()
