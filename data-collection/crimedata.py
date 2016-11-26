import requests
import csv
import os


off_lookup = {
    1: 'Homicide',
    8: 'Assault',
    14: 'Robbery',
    17: 'Other Offences Against the Person',
    21: 'Unlawful Entry',
    27: 'Arson',
    28: 'Other Property Damage',
    29: 'Unlawful Use of Motor Vehicle',
    30: 'Other Theft',
    35: 'Fraud',
    39: 'Handling Stolen Goods',
    45: 'Drug Offence',
    47: 'Liquor (excl. Drunkenness)',
    51: 'Weapons Act Offences',
    52: 'Good Order Offence',
    54: 'Traffic and Related Offences',
    55: 'Other'
}

qld_post_codes = ['4000', '4005', '4006', '4007', '4008', '4009', '4010',
                  '4011', '4012', '4013', '4014', '4017', '4018', '4025',
                  '4030', '4031', '4032', '4034', '4036', '4051', '4053',
                  '4054', '4059', '4060', '4061', '4064', '4065', '4066',
                  '4067', '4068', '4069', '4070', '4073', '4074', '4075',
                  '4076', '4077', '4078', '4101', '4102', '4103', '4104',
                  '4105', '4106', '4107', '4108', '4109', '4110', '4111',
                  '4112', '4113', '4115', '4116', '4120', '4121', '4122',
                  '4151', '4152', '4153', '4154', '4155', '4156', '4169',
                  '4170', '4171', '4172', '4173', '4174', '4178', '4179']


PATH = '.'

def postcode_data(post_code):
    data = requests.get('https://data.police.qld.gov.au/api/qpsmeshblock?boundarylist=' + post_code +
                        '&startdate=659575847&enddate=1372415023&offences=1,8,14,17,21,27,28,29,30,35,39,45,47,51,52,54,55').json()
    offenses = [mesh['OffenceInfo'] for mesh in data['Result']]
    off_sub = {}
    off_sub_date = {}
    dates = set()

    for mesh in offenses:
        for off in mesh:
            if off['Suburb'] in off_sub:
                dates.add(''.join(off['StartDate'].split('-')[:2]))
                off_sub[off['Suburb']].append((
                    off['QpsOffenceId'],
                    off_lookup[off['QpsOffenceCode']],
                    off['StartDate'],
                    off['Solved']))
            else:
                dates.add(''.join(off['StartDate'].split('-')[:2]))
                off_sub[off['Suburb']] = [(
                    off['QpsOffenceId'],
                    off_lookup[off['QpsOffenceCode']],
                    off['StartDate'],
                    off['Solved'])]
            if off['Suburb'] not in off_sub_date:
                off_sub_date[off['Suburb']] = {}

    with open(post_code + 'crimedata.csv', 'w') as f:
        wr = csv.writer(f, delimiter=',')
        print(off_sub.keys())
        for sub in off_sub.keys():
            for date in dates:
                off_sub_date[sub][date] = [0, 0]
            for off in off_sub[sub]:
                d = ''.join(off[2].split('-')[:2])
                off_sub_date[sub][d] = [x + 1 for x in off_sub_date[sub][d]]

        print(off_sub_date.keys())
        for sub in off_sub_date:
            wr.writerow([sub])
            for date in off_sub_date[sub]:
                a = [int(date)]
                a.extend(off_sub_date[sub][date])
                print(a)
                wr.writerow(a)


comp_docs = set()
files = [f for f in os.listdir('.') if os.path.isfile(f) and os.stat(f).st_size and f[-1] == 'v']
for post in range(0, 5000):
    if ('2_' + str(post) + 'crimedata') not in files:
        print(post)
        postcode_data('2_' + str(post))
