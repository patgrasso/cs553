# Patrick Grasso
# I pledge my honor that I have abided by the Stevens Honor System
# 9/17/16

import codecs
import csv

data = csv.reader(codecs.open("2016VAERSDATA.csv", "r", "latin1"))
keys = next(data)
serious_keys = list(map(lambda key: keys.index(key), ["DISABLE", "DIED", "ER_VISIT", "HOSPITAL"]))
writer = csv.writer(open("2016-vaers-serious.csv", "w"))

keys += ["SERIOUS"]
writer.writerow(keys)

for row in data:
    is_serious = False
    for key in serious_keys:
        if row[key] == "Y":
            is_serious = True
    row += ["Y" if is_serious else "N"]
    writer.writerow(row)

