from google.transit import gtfs_realtime_pb2
import google.protobuf.json_format as json_format
import os
from collections import defaultdict
import json
from datetime import datetime
import argparse
import logging

# Parse collected data from our scraper into easily readable and structured JSON files organized by route and trip, with all vehicle positions sorted chronologically. Also checks for duplicate trip IDs within each Protobuf file to ensure data integrity. Used for initial data processing for this project.

# Set up argument parser
parser = argparse.ArgumentParser(description='Process vehicle positions data')
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='Increase verbosity level (-v, -vv, -vvv)')
args = parser.parse_args()

# Set up logging based on verbosity
if args.verbose == 0:
    log_level = logging.WARNING
elif args.verbose == 1:
    log_level = logging.INFO
elif args.verbose == 2:
    log_level = logging.DEBUG
else:
    log_level = logging.DEBUG

logging.basicConfig(level=log_level, format='%(message)s')
logger = logging.getLogger(__name__)


def no_duplicates_pb(file_path):
    """This function checks for duplicate trip_ids in a given Protobuf file"""

    # Initialize the FeedMessage object
    feed = gtfs_realtime_pb2.FeedMessage()

    try:
        # file_path = os.path.join(base_dir, file_path)
        with open(file_path, "rb") as f:
            feed.ParseFromString(f.read())

        # Convert the Protobuf message to a formatted JSON string
        # 'indent=4' makes it look "pretty" and readable
        # pretty_output = json_format.MessageToJson(feed, indent=4)
        # with (open("sample_output.json", "w")) as f:
        #     f.write(pretty_output)
        # print(pretty_output)

        tripIds = set()
        # print("num entries: ", len(feed.entity))
        for item in feed.entity:
            # filter off-duty vehicles
            if item.HasField("vehicle") and item.vehicle.HasField("trip"):
                tripId = item.vehicle.trip.trip_id
                if tripId in tripIds:
                    print(f"Duplicate trip_id found: {tripId}")
                    return False
                tripIds.add(tripId)

        # print(f"--- Decoded Content of {file_path} ---")
        # print(pretty_output)

    except Exception as e:
        print(f"Error: Could not decode the file. {e}")
        exit(1)
    return True


def convert_pb_to_json_file(file_path):
    """This function converts a pb file to a JSON file for easier debugging and inspection."""

    # Initialize the FeedMessage object
    feed = gtfs_realtime_pb2.FeedMessage()

    try:
        with open(file_path, "rb") as f:
            feed.ParseFromString(f.read())

        # Convert the Protobuf message to a formatted JSON string
        pretty_output = json_format.MessageToJson(feed, indent=4)
        with open(f"debug_jsons/{os.path.basename(file_path)}.json", "w") as f:
            f.write(pretty_output)

    except Exception as e:
        print(f"Error: Could not decode the file. {e}")
        exit(1)


def process_vehicle_positions(day):
    """This function processes all Protobuf files for a given day, extracts relevant information, and saves it in a structured JSON format. It organizes the data by route, then by trip, and finally by vehicle positions, ensuring that all entries are sorted chronologically. The resulting JSON is saved as "processed.txt" in the same directory as the original Protobuf files."""
    BASE_DIR = "vehicle_positions_data"
    directory = os.path.join(BASE_DIR, day)
    routes = {}
    tripCounter = 0
    switchCounter = 0
    for filename in sorted(os.listdir(directory)):
        # print(f"Processing file: {filename}")
        if filename.endswith(".pb"):
            file_path = os.path.join(directory, filename)
            assert no_duplicates_pb(
                file_path), f"Duplicate trip_id found in {filename}"

            feed = gtfs_realtime_pb2.FeedMessage()
            with open(file_path, "rb") as f:
                feed.ParseFromString(f.read())

            formatted_time = datetime.fromtimestamp(
                int(feed.header.timestamp)).strftime("%Y_%m_%d_%H_%M_%S")

            for item in feed.entity:
                if item.HasField("vehicle") and item.vehicle.HasField("trip"):
                    routeId = item.vehicle.trip.route_id
                    if routeId not in routes:
                        routes[routeId] = {}
                        routes[routeId]["routeId"] = routeId
                        routes[routeId]["trips"] = {}

                    tripId = item.vehicle.trip.trip_id
                    if tripId not in routes[routeId]["trips"]:
                        tripCounter += 1
                        routes[routeId]["trips"][tripId] = {
                            "tripId": tripId,
                            "startTime": item.vehicle.trip.start_time,
                            "directionId": item.vehicle.trip.direction_id,
                            "vehicleIds": [item.vehicle.vehicle.id],
                            "vehiclePositions": []
                        }

                    vehicleId = item.vehicle.vehicle.id
                    if vehicleId not in routes[routeId]["trips"][tripId]["vehicleIds"]:
                        logger.info(
                            f"Notice: Trip ID {tripId} in route {routeId} has reassigned vehicle ID from {routes[routeId]['trips'][tripId]['vehicleIds'][-1]} to {vehicleId} on {day} at timestamp {formatted_time}.")
                        routes[routeId]["trips"][tripId]["vehicleIds"].append(
                            vehicleId)
                        switchCounter += 1

                    routes[routeId]["trips"][tripId]["vehiclePositions"].append({
                        "latitude": item.vehicle.position.latitude,
                        "longitude": item.vehicle.position.longitude,
                        "bearing": item.vehicle.position.bearing,
                        "speed": item.vehicle.position.speed,
                        "timestamp": item.vehicle.timestamp
                    })

    logger.info(f"Total unique trips processed for {day}: {tripCounter}")
    logger.info(
        f"Total vehicle ID switches detected for {day}: {switchCounter}")
    logger.info(
        f"Ratio of vehicle ID switches to total trips for {day}: {switchCounter/tripCounter:.2%}")

    for route_id in routes:
        # Convert the dictionary of trips into a list of trip objects
        trips_list = list(routes[route_id]["trips"].values())

        # Sort the list by startTime
        trips_list.sort(key=lambda x: x["startTime"])

        # Reassign back to the route (changes the structure from {} to [])
        routes[route_id]["trips"] = trips_list

        for trip in routes[route_id]["trips"]:
            # Sort the vehicle positions by timestamp
            trip["vehiclePositions"].sort(key=lambda x: x["timestamp"])

    pretty_json = json.dumps(routes, indent=4, sort_keys=True)
    # write to json file in directory
    with open(f"{directory}/processed.json", "w") as f:
        f.write(pretty_json)


for day in sorted(os.listdir("vehicle_positions_data")):
    if os.path.isdir(os.path.join("vehicle_positions_data", day)):
        print(f"Processing data for day: {day}")
        process_vehicle_positions(day)
        print(f"Finished processing for day: {day}\n")
