import argparse
import os
import csv
from influxdb import InfluxDBClient
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-db", "--database", help="Database name", default="pinova", nargs='?')
parser.add_argument("-ip", "--hostname", help="Database address (ip/url)", default="161.53.19.199", nargs='?')
parser.add_argument("-p", "--port", help="Database port", default="8086", nargs='?')
parser.add_argument("-f", "--filter", help="List of columns to filter", default='', nargs='?')
args = parser.parse_args()

host = args.hostname
port = args.port
dbname = args.database
filtered_str = args.filter
filtered = [x.strip() for x in filtered_str.split(',')]
client = InfluxDBClient(host, port, database='pinova')

query = 'show measurements'
result = client.query(query)
for measurements in result:
        for measure in measurements:
            measure_name = measure['name']
            query = 'show field keys from "' + measure_name + '"'
            names = ['time', 'readable_time', 'station_id', 'station_name']
            fields_result = client.query(query)
            for field in fields_result:
                for pair in field:
                    name = pair['fieldKey']
                    if name in filtered: continue
                    names.append(name)
                #path = "C:/Users/dkrek/Desktop/"
                    filename = "Pinova/" + measure_name + '.csv'

                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as file:
                    writer = csv.DictWriter(file, names, delimiter=',', lineterminator='\n', extrasaction='ignore')
                    writer.writeheader()
                    query = 'select * from "{}" '.format(measure_name)
                    result = client.query(query, epoch='ms')
                    print(result)
                    for point in result:
                        for item in point:
                            ms = item['time'] / 1000
                            d = datetime.fromtimestamp(ms)
                            item['readable_time'] = d.isoformat(' ')
                            writer.writerow(item)
