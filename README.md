# Synthesis of structured data
Examples of artificial generation of structured data sets based on pre-defined rules as well as synthesis of already existing structured data sets. All examples are written in python.

## Artificial generation
The [Faker](https://github.com/joke2k/faker) python package is used to generate artificial data sets. With this package, it is possible to generate a data set of any size based on pre-defined rules.
There are three examples located inside the "Artificial generation examples" folder.

### Faker_Elektronischer_Impfpass.py
**Background:** 
This script creates an artificial version of the austrian central vaccination register (ELGA). The generated data includes patient identification data and vaccine information. The patient information included covers everything from name, date of birth, gender and address to national insurance number. As well as trade name, classification, manufacturer, expiry date and administration date of the vaccine itself. The number of entries created can be adjusted within the script.

**Requirements:**
- Python version: 3.8+
- Faker version: 14.2+
- Pandas version: 1.5+
- "Reflist_.csv" (List of Austrian postcodes and cities. It can be downloaded from the [Austrian Federal Environment Agency](https://secure.umweltbundesamt.at/edm_portal/redaList.do?seqCode=598c5vaxpkprtj))

**Usage:**
```
python3 Faker_Elektronischer_Impfpass.py
```


### Faker_Jö_Bonusclub.py
This script creates an artificial data set based on data collected by the customer loyalty program "Jö Bonus Club". The records includes selected member information - such as name, gender and address - as well as purchase information - such as product description, place of purchase and accumulated bonus points. The number of entries created can be adjusted within the script.

**Requirements:**
- Python version: 3.8+
- Faker version: 14.2+
- Pandas version: 1.5+
- "Billa_Onlineshop_Produkte.csv" (Contains several products and their prices)

**Usage:**
```
python3 Faker_Jö_Bonusclub.py
```


### Faker_Gesundheitsbefragung.py
This script generates a medical dataset based on the [Austrian Health Survey of 2019](https://www.statistik.at/fileadmin/publications/Oesterreichische-Gesundheitsbefragung2019_Hauptergebnisse.pdf). The data generated contains information on whether a person has high blood pressure, neck pain or back pain. The number of entries created can be adjusted within the script.

**Requirements:**
- Python version: 3.8+
- Faker version: 14.2+

**Usage:**
```
python3 Faker_Gesundheitsbefragung.py
```


## Data synthesis

