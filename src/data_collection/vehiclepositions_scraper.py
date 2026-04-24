from google.transit import gtfs_realtime_pb2
import google.protobuf.json_format as json_format

import requests
import json
import pandas as pd
import os
from collections import defaultdict
from datetime import datetime
from time import sleep

# Scrape vehicle positions every 5 seconds and save raw data to a folder with timestamped filenames based on the internal timestamp in the data. Used for initial data collection for this project.
PB_URL = "https://data.texas.gov/download/eiei-9rpf/application%2Foctet-stream"

print(f"Starting downloader. Press Ctrl+C to stop.")

try:
    while True:
        try:
            # 1. Download the binary data into memory
            response = requests.get(PB_URL, timeout=10)

            if response.status_code == 200:
                # 2. Parse the binary content IMMEDIATELY
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(response.content)

                # 3. Extract the header timestamp
                header_ts = feed.header.timestamp
                readable_ts = datetime.fromtimestamp(
                    header_ts).strftime("%Y_%m_%d_%H_%M_%S")

                today = datetime.now().strftime("%Y_%m_%d")
                folder_path = os.path.join("vehicle_positions_data", today)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # 4. Use that internal timestamp for the filename
                # This ensures your files are named after the ACTUAL data time, not your PC time
                filename = f"positions_{readable_ts}.pb"
                file_path = os.path.join(folder_path, filename)

                # 5. Check if we already have this data to avoid duplicates
                if os.path.exists(file_path):
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Data at {readable_ts} is already saved. Skipping...")
                else:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Saved NEW data from: {readable_ts}")
            else:
                print(f"Failed download. Status: {response.status_code}")

        except Exception as e:
            print(f"An error occurred: {e}")

        sleep(5)

except KeyboardInterrupt:
    print("\nDownloader stopped.")
