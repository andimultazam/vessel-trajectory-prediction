from matplotlib import pyplot as plt
import pandas as pd
import numpy as np 
import csv
from collections import Counter

# with open('ais-processed-2019-12.csv') as csv_file:
#     csv_reader = csv.DictReader(csv_file)

#     shiptype_ctr = Counter()

#     for row in csv_reader:
#         shiptype_ctr.update(row['shiptype'])

def map_shiptype(number):
    if type(number) == str:
        return number
    ship_cat = ''
    first_digit = int(number) // 10
    if first_digit == 1:
        ship_cat = 'Reserved'
    elif first_digit == 2:
        ship_cat = 'Wing in Ground'
    elif first_digit == 3:
        second_digit = int(number) % 10
        if second_digit == 0:
            ship_cat = 'Fishing'
        elif (second_digit == 1 or second_digit == 2):
            ship_cat = 'Tug'
        elif second_digit == 3:
            ship_cat = 'Dredger'
        elif second_digit == 4:
            ship_cat = 'Dive Vessel'
        elif second_digit == 5:
            ship_cat = 'Military Ops'
        elif second_digit == 6:
            ship_cat = 'Sailing Vessel'
        elif second_digit == 7:
            ship_cat = 'Pleasure Craft'
        else:
            ship_cat = 'Reserved'
    elif first_digit == 4:
        ship_cat = 'High-Speed Craft'
    elif first_digit == 5:
        ship_cat = 'SAR'
    elif first_digit == 6:
        ship_cat = 'Passenger'
    elif first_digit == 7:
        ship_cat = 'Cargo'
    elif first_digit == 8:
        ship_cat = 'Tanker'
    else:
        ship_cat = 'Others'
    return ship_cat

df = pd.read_csv('ais-processed-2019-12.csv')
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
df.drop(columns=['Unnamed: 0'], inplace=True)
print("This dataset contains {} records".format(len(df)))

# fill-up shiptype column for each ship by getting them from the same ships in other messages
df_with_stype = df.loc[df.shiptype.notnull()]
df_no_stype = df.loc[df.shiptype.isnull()]
print('total: {}'.format(len(df_with_stype)+len(df_no_stype)))
print('ship with shiptype: {}, ship without shiptype: {}'.format(len(df_with_stype), len(df_no_stype)))

mmsi_with_stype = df_with_stype['mmsi'].value_counts().index.to_list()
mmsi_no_stype = df_no_stype['mmsi'].value_counts().index.to_list()
intersections_stype = list(set(mmsi_with_stype) & set(mmsi_no_stype))
print('There are {} ships who can share the shiptype'.format(len(intersections_stype)))

df.reset_index(inplace=True)
df.set_index('mmsi', inplace=True)

df.loc[:, 'shiptype'] = "Undefined"
for mmsi in intersections_stype:
    if (isinstance(df_with_stype.loc[mmsi, 'shiptype'].tolist(), list)):
        shiptype = list(set(df_with_stype.loc[mmsi, 'shiptype'].tolist()))[0]
    else:
        shiptype = df_with_stype.loc[mmsi, 'shiptype'].tolist()
    df.loc[mmsi, 'shiptype'] = shiptype

# map shiptype to string name
df['shiptype_name'] = df.shiptype.apply(map_shiptype)

def map_status(number):
    status_cat = ''
    if number == 0:
        status_cat = 'Under way'
    elif number == 1:
        status_cat = 'At anchor'
    elif number == 2:
        status_cat = 'Not under command'
    elif number == 3:
        status_cat = 'Restricted'
    elif number == 4:
        status_cat = 'Constrained by Draught'
    elif number == 5:
        status_cat = 'Moored'
    elif number == 6:
        status_cat = 'Aground'
    elif number == 7:
        status_cat = 'Fishing'
    elif number == 8:
        status_cat = 'Sailing'
    elif number == 15:
        status_cat = 'Undefined'
    else:
        status_cat = 'Reserved'
    return status_cat

if __name__ == "__main__":
    pass