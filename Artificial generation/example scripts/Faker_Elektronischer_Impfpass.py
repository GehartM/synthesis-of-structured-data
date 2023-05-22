import faker
from collections import OrderedDict, defaultdict
import csv
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import sys

austrian_postal_code_file_path = 'Reflist_.csv'     # Can be downloaded from https://secure.umweltbundesamt.at/edm_portal/redaList.do?seqCode=598c5vaxpkprtj
vaccination_booster_percentage = 30                 # Percentage of people who had a vaccination booster in the given period
covid_vaccination_booster_percentage = 80           # Percentage of people who had a covid vaccination booster in the given period
faker.proxy.DEFAULT_LOCALE = 'de_AT'
faker_instance = faker.Faker()

state_postal_codes = defaultdict(list)
postal_code_city_names = dict()


class Person:
    def __init__(self):
        self.gender = self.__generate_gender()
        self.first_name = self.__generate_first_name()
        self.last_name = self.__generate_last_name()
        self.age_kategory = str()
        self.age = self.__generate_age()
        self.birthday = self.__generate_birthday()
        self.social_security_number = self.__generate_social_security_number()
        self.state = self.__generate_state()
        self.postal_code = self.__generate_postal_code()
        self.city = self.__generate_city()
        self.street = self.__generate_street()
        self.vaccination_tetanus_diphtheria_polio = self.__generate_vaccination_tetanus_diphtheria_polio()
        self.vaccination_tetanus_diphtheria = self.__generate_vaccination_tetanus_diphtheria()
        self.vaccination_tbe = self.__generate_vaccination_tbe()
        self.vaccination_covid = self.__generate_vaccination_covid()

    def __iter__(self):
        return iter([self.first_name, self.last_name, self.gender, self.birthday, self.social_security_number, self.state, self.postal_code, self.city, self.street,
                     self.vaccination_tetanus_diphtheria.is_vaccinated, self.vaccination_tetanus_diphtheria.name, self.vaccination_tetanus_diphtheria.manufacturer, self.vaccination_tetanus_diphtheria.vaccination_date, self.vaccination_tetanus_diphtheria.expiration_date,
                     self.vaccination_tetanus_diphtheria_polio.is_vaccinated, self.vaccination_tetanus_diphtheria_polio.name, self.vaccination_tetanus_diphtheria_polio.manufacturer, self.vaccination_tetanus_diphtheria_polio.vaccination_date, self.vaccination_tetanus_diphtheria_polio.expiration_date,
                     self.vaccination_tbe.is_vaccinated, self.vaccination_tbe.name, self.vaccination_tbe.manufacturer, self.vaccination_tbe.vaccination_date, self.vaccination_tbe.expiration_date,
                     self.vaccination_covid.is_vaccinated, self.vaccination_covid.name, self.vaccination_covid.manufacturer, self.vaccination_covid.vaccination_date, self.vaccination_covid.expiration_date])

    def __generate_gender(self):
        return faker_instance.random_choices(elements=OrderedDict([("Männlich", 0.4929), ("Weiblich", 0.5071)]), length=1)[0]

    def __generate_first_name(self):
        if self.gender == "Männlich":
            return faker_instance.first_name_male()
        else:
            return faker_instance.first_name_female()

    def __generate_last_name(self):
        return faker_instance.last_name()

    def __generate_age(self):
        if self.gender == "Männlich":
            age_kategory = faker_instance.random_choices(elements=OrderedDict(
                [("15-29", 0.2095),
                 ("30-44", 0.2457),
                 ("45-59", 0.2629),
                 ("60-74", 0.1906),
                 ("75-100", 0.0913)
                 ]), length=1)[0]
            self.age_kategory = age_kategory
            age = age_kategory.split("-")
            return faker_instance.random_int(min=int(age[0]), max=int(age[1]))
        else:
            age_kategory = faker_instance.random_choices(elements=OrderedDict(
                [("15-29", 0.1886),
                 ("30-44", 0.2287),
                 ("45-59", 0.2537),
                 ("60-74", 0.2001),
                 ("75-100", 0.1289)
                 ]), length=1)[0]
            self.age_kategory = age_kategory
            age = age_kategory.split("-")
            return faker_instance.random_int(min=int(age[0]), max=int(age[1]))

    def __generate_birthday(self):
        start = datetime.date(2022, 1, 1) - relativedelta(years=self.age)
        end = datetime.date(2022, 12, 31) - relativedelta(years=self.age)
        print('\nstart date: {}\nend date: {}\nage: {}'.format(start, end, self.age))

        try:
            return faker_instance.date_between_dates(date_start=start, date_end=end)
        except OSError as error:
            print(error)
            print('\nerror: start date: {}\nend date: {}\nage: {}'.format(start, end, self.age))
            sys.exit(1)

    def __generate_social_security_number(self):
        sequence_and_control_number = faker_instance.pyint(min_value=1000, max_value=9999)
        birthday = self.birthday.strftime("%d%m%y")
        return int(str(sequence_and_control_number) + str(birthday))

    def __generate_state(self):
        return faker_instance.random_choices(elements=OrderedDict(
            [("Burgenland", 0.0331),
             ("Kärnten", 0.0629),
             ("Niederösterreich", 0.1892),
             ("Oberösterreich", 0.1676),
             ("Salzburg", 0.0627),
             ("Steiermark", 0.1395),
             ("Tirol", 0.0851),
             ("Vorarlberg", 0.0447),
             ("Wien", 0.2151)
             ]), length=1)[0]

    def __generate_postal_code(self):
        return faker_instance.random_choices(elements=(state_postal_codes[self.state]), length=1)[0]

    def __generate_city(self):
        return faker_instance.random_choices(elements=(postal_code_city_names[self.postal_code]), length=1)[0].strip()

    def __generate_street(self):
        return faker_instance.street_address()

    def __generate_vaccination_tetanus_diphtheria_polio(self):
        if self.gender == "Männlich":
            return Polio(faker_instance.boolean(chance_of_getting_true=(59*(vaccination_booster_percentage/100))))
        else:
            return Polio(faker_instance.boolean(chance_of_getting_true=(59*(vaccination_booster_percentage/100))))

    def __generate_vaccination_tetanus_diphtheria(self):
        if self.vaccination_tetanus_diphtheria_polio.is_vaccinated == 1:
            return Tetanus_Diphtheria(is_vaccinated=True, name=self.vaccination_tetanus_diphtheria_polio.name, manufacturer=self.vaccination_tetanus_diphtheria_polio.manufacturer, vaccination_date=self.vaccination_tetanus_diphtheria_polio.vaccination_date, expiration_date=self.vaccination_tetanus_diphtheria_polio.expiration_date)
        elif self.gender == "Männlich":
            return Tetanus_Diphtheria(faker_instance.boolean(chance_of_getting_true=(70*(vaccination_booster_percentage/100))))
        else:
            return Tetanus_Diphtheria(faker_instance.boolean(chance_of_getting_true=(67*(vaccination_booster_percentage/100))))

    def __generate_vaccination_tbe(self):
        if self.gender == "Männlich":
            return Tbe(faker_instance.boolean(chance_of_getting_true=(62*(vaccination_booster_percentage/100))))
        else:
            return Tbe(faker_instance.boolean(chance_of_getting_true=(63*(vaccination_booster_percentage/100))))

    def __generate_vaccination_covid(self):
        if self.gender == "Männlich":
            if self.age <= 24: return Covid(faker_instance.boolean(chance_of_getting_true=(42*(covid_vaccination_booster_percentage/100))))
            elif self.age <= 34: return Covid(faker_instance.boolean(chance_of_getting_true=(49*(covid_vaccination_booster_percentage/100))))
            elif self.age <= 44: return Covid(faker_instance.boolean(chance_of_getting_true=(55*(covid_vaccination_booster_percentage/100))))
            elif self.age <= 54: return Covid(faker_instance.boolean(chance_of_getting_true=(63*(covid_vaccination_booster_percentage/100))))
            elif self.age <= 64: return Covid(faker_instance.boolean(chance_of_getting_true=(73*(covid_vaccination_booster_percentage/100))))
            elif self.age <= 74: return Covid(faker_instance.boolean(chance_of_getting_true=(80*(covid_vaccination_booster_percentage/100))))
            else: return Covid(faker_instance.boolean(chance_of_getting_true=(85*(covid_vaccination_booster_percentage/100))))
        else:
            if self.age <= 24: return Covid(faker_instance.boolean(chance_of_getting_true=(46*(covid_vaccination_booster_percentage/100))))
            elif self.age <= 34: return Covid(faker_instance.boolean(chance_of_getting_true=(51*(covid_vaccination_booster_percentage/100))))
            elif self.age <= 44: return Covid(faker_instance.boolean(chance_of_getting_true=(56*(covid_vaccination_booster_percentage/100))))
            elif self.age <= 54: return Covid(faker_instance.boolean(chance_of_getting_true=(65*(covid_vaccination_booster_percentage/100))))
            elif self.age <= 64: return Covid(faker_instance.boolean(chance_of_getting_true=(73*(covid_vaccination_booster_percentage/100))))
            elif self.age <= 74: return Covid(faker_instance.boolean(chance_of_getting_true=(79*(covid_vaccination_booster_percentage/100))))
            else: return Covid(faker_instance.boolean(chance_of_getting_true=(81*(covid_vaccination_booster_percentage/100))))


class Vaccination:
    def __init__(self, possible_name_and_manufacturer, is_vaccinated=True, name=None, manufacturer=None, vaccination_date=None, expiration_date=None):
        self.possible_name_and_manufacturer = possible_name_and_manufacturer
        self.is_vaccinated = int(is_vaccinated)
        self.name = name if is_vaccinated is True else None
        self.manufacturer = manufacturer if is_vaccinated is True else None
        self.vaccination_date = vaccination_date if is_vaccinated is True else None
        self.expiration_date = expiration_date if is_vaccinated is True else None
        if is_vaccinated:
            if self.name is None: self.name = self.__generate_name()
            if self.manufacturer is None: self.manufacturer = self.__generate_manufacturer()
            if self.vaccination_date is None: self.vaccination_date = self.__generate_vaccination_date()
            if self.expiration_date is None: self.expiration_date = self.__generate_expiration_date()

    def __iter__(self):
        return iter([self.is_vaccinated, self.name, self.manufacturer, self.expiration_date, self.vaccination_date])

    def __generate_name(self):
        return faker_instance.random_choices(elements=(self.possible_name_and_manufacturer.keys()), length=1)[0]

    def __generate_manufacturer(self):
        return self.possible_name_and_manufacturer[self.name]

    def __generate_vaccination_date(self):
        start = datetime.date(2022, 1, 1)
        end = datetime.date(2022, 12, 31)
        return faker_instance.date_between_dates(date_start=start, date_end=end)

    def __generate_expiration_date(self):
        start = self.vaccination_date + relativedelta(months=1)
        end = self.vaccination_date + relativedelta(years=3)
        return faker_instance.date_between_dates(date_start=start, date_end=end)


class Tetanus_Diphtheria(Vaccination):
    possible_name_and_manufacturer = {
        "Boostrix": "GSK Pharma GmbH",
        "Covaxis": "Sanofi Pasteur Europe",
        "diTeBooster": "AJ Vaccines A/S",
        "DTaP Booster": "AJ Vaccines A/S",
        "dT-reduct Merieux": "Sanofi Pasteur Europe",
        "Td - pur": "Astro Pharma"
    }

    def __init__(self, is_vaccinated, name=None, manufacturer=None, vaccination_date=None, expiration_date=None):
        super().__init__(self.possible_name_and_manufacturer, is_vaccinated=is_vaccinated, name=name, manufacturer=manufacturer, vaccination_date=vaccination_date, expiration_date=expiration_date)


class Polio(Vaccination):
    possible_name_and_manufacturer = {
        "Boostrix Polio": "GSK Pharma GmbH",
        "Repevax": "Sanofi Pasteur Europe",
        "Revaxis": "Sanofi Pasteur Europe"
    }

    def __init__(self, is_vaccinated, name=None, manufacturer=None, vaccination_date=None, expiration_date=None):
        super().__init__(self.possible_name_and_manufacturer, is_vaccinated=is_vaccinated, name=name, manufacturer=manufacturer, vaccination_date=vaccination_date, expiration_date=expiration_date)


class Tbe(Vaccination):
    possible_name_and_manufacturer = {
        "Encepur 0,5 ml": "Bavarian Nordic A/S",
        "FSME-Immun 0,5 ml": "Pfizer Corporation Austria Ges.m.b.H"
    }

    def __init__(self, is_vaccinated, name=None, manufacturer=None, vaccination_date=None, expiration_date=None):
        super().__init__(self.possible_name_and_manufacturer, is_vaccinated=is_vaccinated, name=name, manufacturer=manufacturer, vaccination_date=faker_instance.date_between_dates(date_start=datetime.date(2022, 1, 1) if vaccination_date is None else vaccination_date, date_end=datetime.date(2022, 4, 30)), expiration_date=expiration_date)


class Covid(Vaccination):
    possible_name_and_manufacturer = {
        "Comirnaty 30 µg": "BioNTech Manufacturing GmbH",
        "Comirnaty Original/Omicron BA.1 (15/15 micrograms)": "BioNTech Manufacturing GmbH",
        "Comirnaty Original/Omicron BA.4 -5 (15/15 micrograms)": "BioNTech Manufacturing GmbH",
        "Covid-19 Impfstoff Valneva": "Valneva Austria GmbH",
        "Jcovden (previously: COVID-19 vaccine Janssen)": "Janssen-Cilag International NV",
        "Nuvaxovid": "Novavax CZ",
        "Spikevax (COVID-19 vaccine Moderna) 0,2 mg/mL": "Moderna Biotech Spain, S.L.",
        "Spikevax (COVID -19 vaccine Moderna) 0,1 mg/mL": "Moderna Biotech Spain, S.L.",
        "Spikevax (COVID -19 vaccine Moderna) 50 µg in Fertigspritze": "Moderna Biotech Spain, S.L.",
        "Spikevax bivalent Original/Omicron BA.1 (50 micrograms/50 micrograms)/mL": "Moderna Biotech Spain, S.L.",
        "Spikevax bivalent Original/Omicron BA.4 - 5 (50 micrograms/50 micrograms)/mL": "Moderna Biotech Spain, S.L.",
        "Vaxzevria (COVID-19 vaccine AstraZeneca)": "AstraZeneca AB",
        "VidPrevtyn Beta": "Sanofi Pasteur"
    }

    def __init__(self, is_vaccinated, name=None, manufacturer=None, vaccination_date=None, expiration_date=None):
        super().__init__(self.possible_name_and_manufacturer, is_vaccinated=is_vaccinated, name=name, manufacturer=manufacturer, vaccination_date=vaccination_date, expiration_date=expiration_date)


def get_postal_codes():
    global state_postal_codes
    postal_code_beginnings = {
        "Burgenland": ['7'],
        "Kärnten": ['90', '91', '92', '93', '94', '95', '96', '97', '98'],
        "Niederösterreich": ['2', '3'],
        "Oberösterreich": ['4'],
        "Salzburg": ['5'],
        "Steiermark": ['8'],
        "Tirol": ['60', '61', '62', '63', '64', '65', '66', '99'],
        "Vorarlberg": ['67', '68', '69'],
        "Wien": ['1']
    }

    if not os.path.exists(austrian_postal_code_file_path):
        print("This script needs the austrian postal code list in order to work\nIt can be downloaded from 'https://secure.umweltbundesamt.at/edm_portal/redaList.do?seqCode=598c5vaxpkprtj'.\nSave it in the current working directory.\n")
        sys.exit(1)
    df = pd.read_csv(austrian_postal_code_file_path, sep=',', encoding="utf-8-sig", usecols=[0,2], names=['postal code', 'cities'])
    for postal_code in df['postal code']:
        postal_code_str = str(postal_code)
        for state in postal_code_beginnings.keys():
            for state_beginning in postal_code_beginnings[state]:
                if postal_code_str.startswith(state_beginning):
                    state_postal_codes[state].append(postal_code)

    for index, row in df.iterrows():
        postal_code_city_names[row['postal code']] = row['cities'].split(',')


def generate_fake_entries():
    # Generate 50.000 entries
    entrys = list()
    for x in range(50000):
        entrys.append(Person())
    return entrys


def save_entries_as_csv(generated_entries):
    with open("Faker_Elektronischer_Impfpass_Output.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Vorname", "Nachname", "Geschlecht", "Geburtsdatum", "Sozialversicherungsnummer", "Bundesland", "Postleitzahl", "Ort", "Straße",
                         "Tetanus und Diphterie Impfung", "Tetanus und Diphterie Impfstoff", "Tetanus und Diphterie Impfstoffhersteller", "Tetanus und Diphterie Impfdatum", "Tetanus und Diphterie Ablaufdatum des Impfstoffs",
                         "Polio Impfung", "Polio Impfstoff", "Polio Impfstoffhersteller", "Polio Impfdatum", "Polio Ablaufdatum des Impfstoffs",
                         "FSME Impfung", "FSME Impfstoff", "FSME Impfstoffhersteller", "FSME Impfdatum", "FSME Ablaufdatum des Impfstoffs",
                         "Covid Impfung", "Covid Impfstoff", "Covid Impfstoffhersteller", "Covid Impfdatum", "Covid Ablaufdatum des Impfstoffs"])
        writer.writerows(generated_entries)


def main():
    get_postal_codes()
    generated_entries = generate_fake_entries()
    save_entries_as_csv(generated_entries)


if __name__ == "__main__":
    main()
