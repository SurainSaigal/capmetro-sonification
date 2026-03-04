from google.transit import gtfs_realtime_pb2
import google.protobuf.json_format as json_format
import os
from collections import defaultdict
import json


def no_duplicates_pb(file_path):
    # Initialize the FeedMessage object
    feed = gtfs_realtime_pb2.FeedMessage()

    try:
        # file_path = os.path.join(base_dir, file_path)
        with open(file_path, "rb") as f:
            feed.ParseFromString(f.read())

        # Convert the Protobuf message to a formatted JSON string
        # 'indent=4' makes it look "pretty" and readable
        pretty_output = json_format.MessageToJson(feed, indent=4)
        with (open("sample_output.json", "w")) as f:
            f.write(pretty_output)
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


# {routeId: {tripId: {vehicleId: position[]}}}
def process_vehicle_positions(day):
    BASE_DIR = "vehicle_positions_data"
    directory = os.path.join(BASE_DIR, day)
    routes = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pb"):
            file_path = os.path.join(directory, filename)
            assert no_duplicates_pb(
                file_path), f"Duplicate trip_id found in {filename}"

            feed = gtfs_realtime_pb2.FeedMessage()
            with open(file_path, "rb") as f:
                feed.ParseFromString(f.read())

            # print(json_format.MessageToJson(feed, indent=4))

            for item in feed.entity:
                if item.HasField("vehicle") and item.vehicle.HasField("trip"):
                    routeId = item.vehicle.trip.route_id
                    if routeId not in routes:
                        routes[routeId] = {}
                        routes[routeId]["routeId"] = routeId
                        routes[routeId]["trips"] = {}

                    tripId = item.vehicle.trip.trip_id
                    if tripId not in routes[routeId]["trips"]:
                        routes[routeId]["trips"][tripId] = {
                            "tripId": tripId,
                            "startTime": item.vehicle.trip.start_time,
                            "directionId": item.vehicle.trip.direction_id,
                            "vehicleId": item.vehicle.vehicle.id,
                            "vehiclePositions": []
                        }

                    vehicleId = item.vehicle.vehicle.id
                    assert vehicleId == routes[routeId]["trips"][tripId][
                        "vehicleId"], f"Mismatch in vehicleId for tripId {tripId} in route {routeId}"

                    routes[routeId]["trips"][tripId]["vehiclePositions"].append({
                        "latitude": item.vehicle.position.latitude,
                        "longitude": item.vehicle.position.longitude,
                        "bearing": item.vehicle.position.bearing,
                        "speed": item.vehicle.position.speed,
                        "timestamp": item.vehicle.timestamp or feed.header.timestamp
                    })

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
    # write to txt file in directory
    with open(f"{directory}/processed.txt", "w") as f:
        f.write(pretty_json)


# process_vehicle_positions("2026_03_03")
no_duplicates_pb(
    "vehicle_positions_data/2026_03_03/positions_2026_03_03_18_48_11.pb")
