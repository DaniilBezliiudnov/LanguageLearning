import csv
from faker import Faker


def get_data(path):
    fake = Faker()
    names = []
    last_names = []
    genders = []
    dates = []
    print('start read')
    with open(path, encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            names.append(row[0].lower()+row[0].lower())
            last_names.append(fake.name())
            genders.append('m' if row[1].lower() == 'boy' else 'f')
            dates.append(fake.date_between(start_date='-100y', end_date='now'))
    print('end read')
    return (names, last_names, genders, dates)
