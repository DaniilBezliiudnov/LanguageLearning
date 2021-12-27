import csv
import names.config as config
from faker import Faker

fake = Faker()
names = []
genders = []
dates = []
print('start read')
with open(config.root_dir + '\\data\\names.csv', encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        names.append(row[0].lower())
        genders.append(row[1].lower())
        dates.append(fake.date_between(start_date='-100y', end_date='now'))
print('end read')