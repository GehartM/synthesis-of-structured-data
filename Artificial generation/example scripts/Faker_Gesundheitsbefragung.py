import faker
from collections import OrderedDict
import csv

faker.proxy.DEFAULT_LOCALE = 'de_AT'
faker_instance = faker.Faker()


class Person:
    def __init__(self):
        self.gender = self.__generate_gender()
        self.first_name = self.__generate_first_name()
        self.last_name = self.__generate_last_name()
        self.age_kategory = str()
        self.age = self.__generate_age()
        self.neck_pain = self.__generate_illness_neck_pain()
        self.high_blood_pressure = self.__generate_illness_high_blood_pressure()
        self.back_pain = self.__generate_illness_back_pain()

    def __iter__(self):
        return iter([self.first_name, self.last_name, self.gender, self.age, self.age_kategory, self.neck_pain, self.back_pain, self.high_blood_pressure])

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

    def __generate_illness_neck_pain(self):
        if self.gender == "Männlich":
            if self.age <= 29: return int(faker_instance.boolean(chance_of_getting_true=1))
            elif self.age <= 44: return int(faker_instance.boolean(chance_of_getting_true=8))
            elif self.age <= 59: return int(faker_instance.boolean(chance_of_getting_true=20))
            elif self.age <= 74: return int(faker_instance.boolean(chance_of_getting_true=19))
            else: return int(faker_instance.boolean(chance_of_getting_true=24))
        else:
            if self.age <= 29: return int(faker_instance.boolean(chance_of_getting_true=11))
            elif self.age <= 44: return int(faker_instance.boolean(chance_of_getting_true=18))
            elif self.age <= 59: return int(faker_instance.boolean(chance_of_getting_true=30))
            elif self.age <= 74: return int(faker_instance.boolean(chance_of_getting_true=29))
            else: return int(faker_instance.boolean(chance_of_getting_true=34))

    def __generate_illness_high_blood_pressure(self):
        if self.gender == "Männlich":
            if self.age <= 29: return int(faker_instance.boolean(chance_of_getting_true=6))
            elif self.age <= 44: return int(faker_instance.boolean(chance_of_getting_true=6))
            elif self.age <= 59: return int(faker_instance.boolean(chance_of_getting_true=23))
            elif self.age <= 74: return int(faker_instance.boolean(chance_of_getting_true=42))
            else: return int(faker_instance.boolean(chance_of_getting_true=50))
        else:
            if self.age <= 29: return int(faker_instance.boolean(chance_of_getting_true=3))
            elif self.age <= 44: return int(faker_instance.boolean(chance_of_getting_true=3))
            elif self.age <= 59: return int(faker_instance.boolean(chance_of_getting_true=23))
            elif self.age <= 74: return int(faker_instance.boolean(chance_of_getting_true=42))
            else: return int(faker_instance.boolean(chance_of_getting_true=56))

    def __generate_illness_back_pain(self):
        if self.gender == "Männlich":
            if self.age <= 29: return int(faker_instance.boolean(chance_of_getting_true=10))
            elif self.age <= 44: return int(faker_instance.boolean(chance_of_getting_true=18))
            elif self.age <= 59: return int(faker_instance.boolean(chance_of_getting_true=32))
            elif self.age <= 74: return int(faker_instance.boolean(chance_of_getting_true=36))
            else: return int(faker_instance.boolean(chance_of_getting_true=33))
        else:
            if self.age <= 29: return int(faker_instance.boolean(chance_of_getting_true=10))
            elif self.age <= 44: return int(faker_instance.boolean(chance_of_getting_true=18))
            elif self.age <= 59: return int(faker_instance.boolean(chance_of_getting_true=32))
            elif self.age <= 74: return int(faker_instance.boolean(chance_of_getting_true=36))
            else: return int(faker_instance.boolean(chance_of_getting_true=50))


def generate_fake_entries():
    # Generate 100 entries
    entrys = list()
    for x in range(100):
        entrys.append(Person())
    return entrys


def save_entries_as_csv(generated_entries):
    with open("Faker_Gesundheitsbefragung_Output.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Vorname", "Nachname", "Geschlecht", "Alter", "Alterskategorie", "Nackenschmerzen", "Rückenschmerzen", "Bluthochdruck"])
        writer.writerows(generated_entries)


def main():
    generated_entries = generate_fake_entries()
    save_entries_as_csv(generated_entries)


if __name__ == "__main__" :
    main()
