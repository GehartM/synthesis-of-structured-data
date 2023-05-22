import faker
from collections import OrderedDict, defaultdict
import csv
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import sys

product_file_path = 'Billa_Onlineshop_Produkte.csv'     # Contains products offered inside the shop
average_food_spending_weekly_per_household = 98
faker.proxy.DEFAULT_LOCALE = 'de_AT'
faker_instance = faker.Faker()

product_list = None


class Person:
    def __init__(self):
        self.membership_number = self.__generate_membership_number()
        self.gender = self.__generate_gender()
        self.first_name = self.__generate_first_name()
        self.last_name = self.__generate_last_name()
        self.age_kategory = str()
        self.age = self.__generate_age()
        self.birthday = self.__generate_birthday()
        self.postal_code = self.__generate_postal_code()
        self.city = 'Wien'
        self.street = self.__generate_street()
        self.telephone_number = self.__generate_telephone_number()
        self.email_address = self.__generate_email_address()

    def __iter__(self):
        return iter([self.membership_number, self.first_name, self.last_name, self.gender, self.age_kategory, self.birthday, self.postal_code, self.city, self.street, self.telephone_number, self.email_address])

    def __generate_membership_number(self):
        return faker_instance.unique.random_int(min=1111111, max=9999999)

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
        try:
            return faker_instance.date_between_dates(date_start=start, date_end=end)
        except OSError as error:
            print(error)
            print('\nstart date: {}\nend date: {}\nage: {}'.format(start, end, self.age))
            sys.exit(1)

    def __generate_postal_code(self):
        return faker_instance.random_choices(elements=(1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1210, 1220, 1230), length=1)[0]

    def __generate_street(self):
        return faker_instance.street_address()

    def __generate_telephone_number(self):
        country_code = faker_instance.random_choices(elements=('+43', '0'), length=1)[0]
        phone_number = faker_instance.random_int(min=61234567890, max=69999999999)
        return country_code + str(phone_number)

    def __generate_email_address(self):
        first_part = faker_instance.random_choices(elements=(self.first_name, self.first_name[0]), length=1)[0]
        second_part = faker_instance.random_choices(elements=(self.last_name, '{}{}'.format(self.last_name, self.birthday.strftime('%y')), '{}{}'.format(self.last_name, faker_instance.random_int(min=1, max=999))), length=1)[0]
        return '{}.{}@{}'.format(first_part, second_part, faker_instance.free_email_domain())


class Purchase:
    def __init__(self):
        self.person = Person()
        self.total_price_limit = self.__generate_total_price_limit()
        self.total_price = 0
        self.collected_bonus_points = 0
        self.products = self.__generate_purchased_products()
        self.purchase_date = self.__generate_purchase_date()
        self.purchase_time = self.__generate_purchase_time()
        self.payment_method = self.__generate_payment_method()

    def __iter__(self):
        return iter([self.person.membership_number, self.person.first_name, self.person.last_name, self.person.gender, self.person.age_kategory, self.person.birthday, self.person.postal_code, self.person.city, self.person.street, self.person.telephone_number, self.person.email_address,
                     self.total_price, self.collected_bonus_points, self.purchase_time, self.payment_method])

    def __generate_total_price_limit(self):
        return faker_instance.random_int(min=int(average_food_spending_weekly_per_household*0.7), max=int(average_food_spending_weekly_per_household*1.3))

    def __generate_purchased_products(self):
        purchased_products = list()
        spent_amount = 0
        while spent_amount < self.total_price_limit:
            purchased_products.append(Product(self.person.age))
            spent_amount += (purchased_products[-1].price * purchased_products[-1].quantity)
        self.total_price = round(spent_amount, 2)
        self.collected_bonus_points = int(spent_amount//2)
        return purchased_products

    def __generate_purchase_date(self):
        return faker_instance.random_choices(elements=OrderedDict(
            [(datetime.date(2022, 3, 14), 0.1),
             (datetime.date(2022, 3, 15), 0.1),
             (datetime.date(2022, 3, 16), 0.1),
             (datetime.date(2022, 3, 17), 0.1),
             (datetime.date(2022, 3, 18), 0.2),
             (datetime.date(2022, 3, 19), 0.3)
             ]), length=1)[0]

    def __generate_purchase_time(self):
        if self.purchase_date.weekday() == 5:
            start = datetime.datetime.combine(self.purchase_date, datetime.time(7, 42, 0))
            end = datetime.datetime.combine(self.purchase_date, datetime.time(18, 0, 0))
        else:
            start = datetime.datetime.combine(self.purchase_date, datetime.time(7, 32, 0))
            end = datetime.datetime.combine(self.purchase_date, datetime.time(19, 50, 0))
        return faker_instance.date_time_between_dates(datetime_start=start, datetime_end=end)

    def __generate_payment_method(self):
        return faker_instance.random_choices(elements=('Barzahlung', 'Bankomatkarte', 'Kreditkarte'), length=1)[0]


class Product:
    def __init__(self, persons_age):
        self.consumption_expenditure_kategorie = self.__generate_consumption_expenditure_kategorie(persons_age)
        self.kategorie = None
        self.price = 0
        self.name = self.__generate_name()
        self.quantity = self.__generate_quantity()

    def __iter__(self):
        return iter([self.consumption_expenditure_kategorie, self.kategorie, self.price, self.name, self.quantity])

    def __generate_consumption_expenditure_kategorie(self, persons_age):
        # Check if the person is allowed to drink alcohol.
        bier_percentage = 0.04
        wine_percentage = 0.04
        if persons_age < 16:
            bier_percentage = 0
            wine_percentage = 0

        # People aged between 15-29 are said to have a higher proportion of sweets
        sweet_percentage = 0.07
        if persons_age <= 29:
            sweet_percentage = 0.14

        return faker_instance.random_choices(elements=OrderedDict(
            [("Obst", 0.08),
             ("Gemüse", 0.10),
             ("Brot, Getreideprodukte", 0.18),
             ("Mineralwasser, Limonaden, Säfte", 0.06),
             ("Alkoholische Getränke auf Weinbasis", wine_percentage),
             ("Bier", bier_percentage),
             ("Eier", 0.02),
             ("Fleisch", 0.11),
             ("Wurst- und Selchwaren", 0.08),
             ("Milchprodukte", 0.11),
             ("Fisch, Meerestiere", 0.03),
             ("Süßwaren", sweet_percentage)
             ]), length=1)[0]

    def __generate_name(self):
        possible_products_df = product_list[product_list['Verbrauchausgaben-Kategorie'] == self.consumption_expenditure_kategorie]
        purchased_product = faker_instance.random_choices(elements=(possible_products_df['Produktname'].tolist()), length=1)[0]
        purchased_product_row = possible_products_df.loc[possible_products_df['Produktname'] == purchased_product]
        self.kategorie = purchased_product_row.iloc[0]['Kategorie']
        self.price = purchased_product_row.iloc[0]['Preis in €']
        return ('{} {}').format(purchased_product, purchased_product_row.iloc[0]['Menge'])

    def __generate_quantity(self):
        return faker_instance.random_choices(elements=OrderedDict(
            [(1, 0.8),
             (2, 0.15),
             (3, 0.05)
             ]), length=1)[0]


def get_product_list():
    global product_list

    if not os.path.exists(product_file_path):
        print("This script needs the file '{}' in order to work.\nIt contains a list of products offered inside a shop.\n".format(product_file_path))
        sys.exit(1)
    product_list = pd.read_csv(product_file_path, sep=';', decimal=',', encoding="utf-8-sig")


def generate_fake_entries():
    # Generate 100 entrys
    entrys = list()
    for x in range(3000):
        entrys.append(Purchase())
    return entrys


def save_entries_as_csv(generated_entries):
    with open("Faker_Jö_Bonusclub_Output.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Produktname", "Kategorie", "Verbrauchausgaben-Kategorie", "Preis in € je Stk", "Anzahl",
                         "Gesamtpreis", "Gesammelte Ös", "Kaufdatum", "Bezahlart",
                         "Mitgliedsnummer", "Vorname", "Nachname", "Geschlecht", "Geburtsdatum", "Postleitzahl", "Ort", "Straße", "Telefonnummer", "E-Mailadresse"])

        row_count = 0
        for purchase in generated_entries:
            for product in purchase.products:
                writer.writerow([product.name, product.kategorie, product.consumption_expenditure_kategorie, str(product.price).replace('.', ','), product.quantity,
                                 str(purchase.total_price).replace('.', ','), purchase.collected_bonus_points, purchase.purchase_time, purchase.payment_method,
                                 purchase.person.membership_number, purchase.person.first_name, purchase.person.last_name,
                                 purchase.person.gender,
                                 purchase.person.birthday,
                                 purchase.person.postal_code,
                                 purchase.person.city, purchase.person.street, purchase.person.telephone_number,
                                 purchase.person.email_address])
                row_count += 1
        print("{} rows were successfully written to csv.".format(row_count))


def main():
    get_product_list()
    generated_entries = generate_fake_entries()
    save_entries_as_csv(generated_entries)


if __name__ == "__main__" :
    main()
