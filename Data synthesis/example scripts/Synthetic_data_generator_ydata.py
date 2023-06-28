from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
import pandas as pd
from collections import namedtuple
import os
import sys
import csv
from pathlib import Path
import datetime
import argparse


input_folder = Path('Input')
output_folder = Path('Output')
epochs = 100
batch_size = 4000
command_line_args = False

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
        return df
    return df


def convert_specific_columns_to_object(input_file, df):
    if input_file == 'Faker_Jö_Bonusclub_Output.csv':
        df['Telefonnummer'] = df['Telefonnummer'].astype(object)
        return df
    return df


def get_numerical_categorical_columns(csv_file_name):
    if csv_file_name == 'Case_Information.csv':
        numerical_columns = ['age', 'date_announced_day', 'date_announced_month', 'date_announced_year', 'date_recovered_day', 'date_recovered_month', 'date_recovered_year', 'date_of_death_day', 'date_of_death_month', 'date_of_death_year', 'date_announced_as_removed_day', 'date_announced_as_removed_month', 'date_announced_as_removed_year', 'date_of_onset_of_symptoms_day', 'date_of_onset_of_symptoms_month', 'date_of_onset_of_symptoms_year']
        categorical_columns = ['case_id', 'sex', 'status', 'province', 'muni_city', 'health_status', 'home_quarantined', 'pregnant', 'region']
    elif csv_file_name == 'Faker_Jö_Bonusclub_Output.csv':
        numerical_columns = ['Preis in € je Stk', 'Anzahl', 'Gesamtpreis', 'Gesammelte Ös', 'Kaufdatum_Tag', 'Kaufdatum_Monat', 'Kaufdatum_Jahr', 'Kaufdatum_Stunde', 'Kaufdatum_Minute', 'Kaufdatum_Sekunde', 'Mitgliedsnummer', 'Geburtsdatum_Tag', 'Geburtsdatum_Monat', 'Geburtsdatum_Jahr']
        categorical_columns = ['Produktname', 'Kategorie', 'Verbrauchausgaben-Kategorie', 'Bezahlart', 'Vorname', 'Nachname', 'Geschlecht', 'Postleitzahl', 'Ort', 'Straße', 'Telefonnummer', 'E-Mailadresse']
    elif csv_file_name == 'Online Retail.csv':
        numerical_columns = ['Quantity', 'InvoiceDate_day', 'InvoiceDate_month', 'InvoiceDate_year', 'InvoiceDate_hour', 'InvoiceDate_minute', 'UnitPrice', 'CustomerID']
        categorical_columns = ['InvoiceNo', 'StockCode', 'Description', 'Country']
    elif csv_file_name == 'Faker_Elektronischer_Impfpass_Output.csv':
        numerical_columns = ['Geburtsdatum_Tag', 'Geburtsdatum_Monat', 'Geburtsdatum_Jahr', 'Sozialversicherungsnummer', 'Tetanus und Diphterie Impfdatum_Tag', 'Tetanus und Diphterie Impfdatum_Monat', 'Tetanus und Diphterie Impfdatum_Jahr', 'Tetanus und Diphterie Ablaufdatum des Impfstoffs_Tag', 'Tetanus und Diphterie Ablaufdatum des Impfstoffs_Monat', 'Tetanus und Diphterie Ablaufdatum des Impfstoffs_Jahr', 'Polio Impfdatum_Tag', 'Polio Impfdatum_Monat', 'Polio Impfdatum_Jahr', 'Polio Ablaufdatum des Impfstoffs_Tag', 'Polio Ablaufdatum des Impfstoffs_Monat', 'Polio Ablaufdatum des Impfstoffs_Jahr', 'FSME Impfdatum_Tag', 'FSME Impfdatum_Monat', 'FSME Impfdatum_Jahr', 'FSME Ablaufdatum des Impfstoffs_Tag', 'FSME Ablaufdatum des Impfstoffs_Monat', 'FSME Ablaufdatum des Impfstoffs_Jahr', 'Covid Impfdatum_Tag', 'Covid Impfdatum_Monat', 'Covid Impfdatum_Jahr', 'Covid Ablaufdatum des Impfstoffs_Tag', 'Covid Ablaufdatum des Impfstoffs_Monat', 'Covid Ablaufdatum des Impfstoffs_Jahr']
        categorical_columns = ['Vorname', 'Nachname', 'Geschlecht', 'Bundesland', 'Postleitzahl', 'Ort', 'Straße', 'Tetanus und Diphterie Impfung', 'Tetanus und Diphterie Impfstoff', 'Tetanus und Diphterie Impfstoffhersteller', 'Polio Impfung', 'Polio Impfstoff', 'Polio Impfstoffhersteller', 'FSME Impfung', 'FSME Impfstoff', 'FSME Impfstoffhersteller', 'Covid Impfung', 'Covid Impfstoff', 'Covid Impfstoffhersteller']
    else:
        print('\n\t[!] Got unsupported csv-file "{}"!'.format(csv_file_name))
        sys.exit(1)

    return numerical_columns, categorical_columns


def synthesise_with_wgan(input_file):
    print('\n\t[-] Starting synthesis of "{}" with WGAN...'.format(input_file))
    df = read_csv_file(input_file)

    if df is not None:
        df = convert_specific_columns_to_boolean(input_file, df)
        df = convert_specific_columns_to_object(input_file, df)
        numerical_columns, categorical_columns = get_numerical_categorical_columns(input_file)

        beta_1 = 0.5
        beta_2 = 0.9

        model_arguments = ModelParameters(batch_size=batch_size, betas=(beta_1, beta_2))
        train_arguments = TrainParameters(epochs=epochs)
        model = RegularSynthesizer(modelname='wgan', model_parameters=model_arguments, n_critic=10)
        print('\t\t[.] Fitting the model...')
        model.fit(data=df, train_arguments=train_arguments, num_cols=numerical_columns, cat_cols=categorical_columns)

        print('\t\t[.] Saving the model...')
        model.save(path='{}.pkl'.format(output_folder / '{}_{}'.format('WGAN', input_file[:-4])))
        sample_count = len(df.index)
        print('\t\t[.] Generating {} samples from fitted model...'.format(sample_count))
        synthetic_df = model.sample(n_samples=sample_count)
        output_file = save_synthetic_data_as_csv(synthetic_df, input_file, 'WGAN')
        print('\t\t[.] {} samples saved to "{}".'.format(sample_count, output_file))


def synthesise_with_wgan_gp(input_file):
    print('\n\t[-] Starting synthesis of "{}" with WGAN-GP...'.format(input_file))
    df = read_csv_file(input_file)

    if df is not None:
        df = convert_specific_columns_to_boolean(input_file, df)
        df = convert_specific_columns_to_object(input_file, df)
        numerical_columns, categorical_columns = get_numerical_categorical_columns(input_file)

        beta_1 = 0.5
        beta_2 = 0.9

        model_arguments = ModelParameters(batch_size=batch_size, betas=(beta_1, beta_2))
        train_arguments = TrainParameters(epochs=epochs)
        model = RegularSynthesizer(modelname='wgangp', model_parameters=model_arguments, n_critic=10)
        print('\t\t[.] Fitting the model...')
        model.fit(data=df, train_arguments=train_arguments, num_cols=numerical_columns, cat_cols=categorical_columns)

        print('\t\t[.] Saving the model...')
        model.save(path='{}.pkl'.format(output_folder / '{}_{}'.format('WGAN-GP', input_file[:-4])))
        sample_count = len(df.index)
        print('\t\t[.] Generating {} samples from fitted model...'.format(sample_count))
        synthetic_df = model.sample(n_samples=sample_count)
        output_file = save_synthetic_data_as_csv(synthetic_df, input_file, 'WGAN-GP')
        print('\t\t[.] {} samples saved to "{}".'.format(sample_count, output_file))


def save_synthetic_data_as_csv(synthetic_df, output_filename, synthetic_method):
    output_filename = output_folder / '{}_{}'.format(synthetic_method, output_filename)
    synthetic_df.to_csv(output_filename, index=False, encoding='utf-8-sig', sep=';', decimal=',')
    return output_filename


Synthetic_method = namedtuple('Synthetic_method', 'name function')
supported_synthetic_methods = [
    Synthetic_method('WGAN', synthesise_with_wgan),
    Synthetic_method('WGAN-GP', synthesise_with_wgan_gp)
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

    for csv_file in csv_files:
        for method in synthetic_methods:
            supported_synthetic_methods[method].function(csv_file)

    sys.stdout = stdout_backup
    log_file_handler.close()


if __name__ == "__main__":
    main()
