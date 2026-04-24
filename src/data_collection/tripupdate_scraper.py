import requests
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
from time import sleep

# Scrape trip updates every 30 seconds and save raw data and route-to-trips mapping to separate folders with timestamped filenames.
tripupdates_url = "https://data.texas.gov/download/mqtr-wwpy/text%2Fplain"

prev_file_timestamp = None
while True:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(f"Fetching data at {timestamp}...")
    try:
        response = requests.get(tripupdates_url)
        response.raise_for_status()

        raw_data = response.json()
        cur_file_timestamp = raw_data.get(
            'header', {}).get('timestamp', 'unknown')
        if cur_file_timestamp == prev_file_timestamp:
            print("No new data since last fetch.")
            sleep(30)
            continue
        prev_file_timestamp = cur_file_timestamp

        formatted_time = datetime.fromtimestamp(
            int(cur_file_timestamp)).strftime("%Y_%m_%d_%H_%M_%S")

        formatted_raw = json.dumps(raw_data, indent=2)
        with open(f'raw_data/{formatted_time}.json', 'w') as f:
            f.write(formatted_raw)

        route_to_trips = defaultdict(list)
        if 'entity' in raw_data:
            for item in raw_data['entity']:
                route_id = item['tripUpdate']['trip']['routeId']
                route_to_trips[route_id].append(item['tripUpdate'])

        formatted_route_to_trips = json.dumps(route_to_trips, indent=2)
        with open(f'route_to_trips/{formatted_time}.txt', 'w') as fout:
            fout.write(formatted_route_to_trips)

    except Exception as e:
        print(f"Error: {e}")
    sleep(30)
