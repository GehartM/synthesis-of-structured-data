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
**Background:** 
This script creates an artificial data set based on data collected by the customer loyalty program "Jö Bonus Club". The records includes selected member information - such as name, gender and address - as well as purchase information - such as product description, place of purchase and accumulated bonus points. The number of entries created can be adjusted within the script.

**Requirements:**
- Python version: 3.8+
- Faker version: 14.2+
- Pandas version: 1.5+
- ["Billa_Onlineshop_Produkte.csv"](https://github.com/GehartM/synthesis-of-structured-data/blob/main/Artificial%20generation/example%20scripts/Billa_Onlineshop_Produkte.csv) (Contains several products and their prices)

**Usage:**
```
python3 Faker_Jö_Bonusclub.py
```


### Faker_Gesundheitsbefragung.py
**Background:** 
This script generates a medical dataset based on the [Austrian Health Survey of 2019](https://www.statistik.at/fileadmin/publications/Oesterreichische-Gesundheitsbefragung2019_Hauptergebnisse.pdf). The data generated contains information on whether a person has high blood pressure, neck pain or back pain. The number of entries created can be adjusted within the script.

**Requirements:**
- Python version: 3.8+
- Faker version: 14.2+

**Usage:**
```
python3 Faker_Gesundheitsbefragung.py
```


## Data synthesis
### Synthetic_data_generator_sdv.py
**Background:**
This script can be used to synthesise any structured data with CTGAN or TVAE. The implementation of synthesis methods from SDV is used for this purpose. The script is optimised for the synthesis of the four data sets [Case_Information.csv](https://www.kaggle.com/datasets/cvronao/covid19-philippine-dataset?select=Case_Information.csv), [Online Retail.csv](https://archive.ics.uci.edu/ml/datasets/online+retail), [Faker_Elektronischer_Impfpass_Output.csv](https://github.com/GehartM/synthesis-of-structured-data/blob/main/Artificial%20generation/example%20output/Faker_Elektronischer_Impfpass_Output.csv) and [Faker_Jö_Bonusclub_Output.csv](https://github.com/GehartM/synthesis-of-structured-data/blob/main/Artificial%20generation/example%20output/Faker_J%C3%B6_Bonusclub_Output.csv).

**Requirements:**
- Python version: 3.8+
- Pandas version: 1.5+
- SDV version: 1.1.0+
- [Custom_constraints_Case_Information.py](https://github.com/GehartM/synthesis-of-structured-data/blob/main/Data%20synthesis/example%20scripts/Custom_constraints_Case_Information.py)
- [Custom_constraints_Elektronischer_Impfpass.py](https://github.com/GehartM/synthesis-of-structured-data/blob/main/Data%20synthesis/example%20scripts/Custom_constraints_Elektronischer_Impfpass.py)
- json metadata file for input csv (the metadata files for "Case_Information.csv", "Online Retail.csv", "Faker_Elektronischer_Impfpass_Output.csv" and "Faker_Jö_Bonusclub_Output.csv" can be found inside the directory [metadata files](https://github.com/GehartM/synthesis-of-structured-data/tree/main/Data%20synthesis/example%20scripts/metadata%20files))

**Usage:**
```
Synthetic_data_generator_sdv.py [-m METHODS]
```

### Beispieldaten_fiktiver_Personen_sdv_ctgan.py
**Background:**
Simple example showing how the CTGAN of SDV and the Faker library works.

**Requirements:**
- Python version: 3.8+
- Pandas version: 1.5+
- SDV version: 0.18
- ["Beispieldaten_fiktiver_Personen.csv"](https://github.com/GehartM/synthesis-of-structured-data/blob/main/Data%20synthesis/example%20scripts/Beispieldaten_fiktiver_Personen.csv)

**Usage:**
```
python3 Beispieldaten_fiktiver_Personen_sdv_ctgan.py
```


### Bayesian_network_example_Gesundheitsbefragung.py
**Background:**
This script serves as an illustration of how a Bayesian network can be trained in order to generate synthetic data. For this purpose, an artificially generated data set based on the [Austrian Health Survey of 2019](https://www.statistik.at/fileadmin/publications/Oesterreichische-Gesundheitsbefragung2019_Hauptergebnisse.pdf) is used as input.

**Requirements:**
- Python version: 3.8+
- Pandas version: 1.5+
- pgmpy version: 0.1.21+
- Matplotlib version: 3.6+
- Seaborn version: 0.12+
- plotly version: 5.13+
- cpt_tools (Can be downloaded from [grahamharrison68](https://gist.github.com/grahamharrison68/1187c53d078c3c899b534852fe8edf9c))
-"Faker_Gesundheitsbefragung_Output.csv" (Can be generated with [Faker_Gesundheitsbefragung.py](https://github.com/GehartM/synthesis-of-structured-data/blob/main/Artificial%20generation/example%20scripts/Faker_Gesundheitsbefragung.py) or download the pre-generated [example](https://github.com/GehartM/synthesis-of-structured-data/blob/main/Artificial%20generation/example%20output/Faker_Gesundheitsbefragung_Output.csv))


**Usage:**
```
python3 Bayesian_network_example_Gesundheitsbefragung.py
```


### DP_example.py
**Background:**
Comparison of aggregated evaluations of data sets with and without differential privacy.

**Requirements:**
- Python version: 3.8+
- Pandas version: 1.5+
- Python-DP version: 1.1.1
- ["Beispieldaten_fiktiver_Personen.csv"](https://github.com/GehartM/synthesis-of-structured-data/blob/main/Data%20synthesis/example%20scripts/Beispieldaten_fiktiver_Personen.csv)

**Usage:**
```
python3 DP_example.py
```
