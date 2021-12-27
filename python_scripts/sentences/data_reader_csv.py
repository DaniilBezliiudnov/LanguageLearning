import csv
import config
data = []
labels = []
with open(config.root_dir + '\\data\\1.csv', encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        labels.append(int(row[0]))
        data.append(row[1].lower())