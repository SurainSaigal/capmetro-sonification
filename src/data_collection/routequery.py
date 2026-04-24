import os
import json

# proof of concept for querying the processed data. Used for initial data exploration for this project.


def get_route_info():
    # get user input
    day = input("Enter day (YYYY_MM_DD): ")
    if not os.path.exists(os.path.join("vehicle_positions_data", day)):
        print(f"No data found for {day}. Please check the date and try again.")
        return

    while True:
        route_id = input("Enter route: ")

        with open(f"vehicle_positions_data/{day}/processed.txt", "r") as f:
            data = json.load(f)
            if route_id not in data:
                print(
                    f"Route {route_id} not found in processed data for day {day}")
                return

            num_trips = len(data[route_id]["trips"])
            northbound_trips = sum(
                1 for trip in data[route_id]["trips"] if trip["directionId"] == 0)
            southbound_trips = sum(
                1 for trip in data[route_id]["trips"] if trip["directionId"] == 1)
            assert num_trips == northbound_trips + \
                southbound_trips, f"Mismatch in trip counts for route {route_id} on {day}"

            print(
                f"Route {route_id} had {num_trips} trips on {day}. {northbound_trips} northbound and {southbound_trips} southbound.")

            print("The earliest trip started at", min(
                trip["startTime"] for trip in data[route_id]["trips"]))
            print("The latest trip started at", max(
                trip["startTime"] for trip in data[route_id]["trips"]))


get_route_info()
