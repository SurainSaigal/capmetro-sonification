import argparse
import json
import os
import webbrowser

import pandas as pd
import pydeck as pdk
from keplergl import KeplerGl


def load_data(date: str) -> pd.DataFrame:
    # pwd
    print(os.getcwd())
    with open(f"../vehicle_positions_data/{date}/processed.json", "r") as f:
        raw_data = json.load(f)

    flat_data = []
    for key, route_info in raw_data.items():
        route_id = route_info.get("routeId")
        for trip in route_info.get("trips", []):
            trip_id = trip.get("tripId")
            vehicle_ids = trip.get("vehicleIds", [])
            vehicle_id = vehicle_ids[0] if vehicle_ids else "Unknown"
            for pos in trip.get("vehiclePositions", []):
                flat_data.append({
                    "route_id": route_id,
                    "trip_id": trip_id,
                    "vehicle_id": vehicle_id,
                    "latitude": pos.get("latitude"),
                    "longitude": pos.get("longitude"),
                    "speed": pos.get("speed"),
                    "bearing": pos.get("bearing"),
                    "raw_timestamp": pos.get("timestamp"),
                })

    df = pd.DataFrame(flat_data)
    df['timestamp'] = (
        pd.to_datetime(df['raw_timestamp'], unit='s')
        .dt.tz_localize('UTC')
        .dt.tz_convert('America/Chicago')
        .dt.strftime('%Y-%m-%d %H:%M:%S')
    )
    df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    return df


def build_kepler_map(df: pd.DataFrame) -> None:
    map_config = {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [
                    {
                        "dataId": ["CapMetro_Buses"],
                        "id": "8oj7szg7q",
                        "name": ["timestamp"],
                        "type": "timeRange",
                        "value": [1772610478810.5244, 1772610594810.5244],
                        "plotType": {
                            "interval": "15-minute",
                            "defaultTimeFormat": "L  LT",
                            "type": "histogram",
                            "aggregation": "sum",
                        },
                        "animationWindow": "free",
                        "yAxis": None,
                        "view": "enlarged",
                        "speed": 0.22,
                        "enabled": True,
                    }
                ],
                "layers": [
                    {
                        "id": "lk8579l",
                        "type": "point",
                        "config": {
                            "dataId": "CapMetro_Buses",
                            "columnMode": "points",
                            "label": "Point",
                            "color": [231, 159, 213],
                            "highlightColor": [252, 242, 26, 255],
                            "columns": {"lat": "latitude", "lng": "longitude"},
                            "isVisible": True,
                            "visConfig": {
                                "radius": 5.5,
                                "fixedRadius": False,
                                "opacity": 0.8,
                                "outline": False,
                                "thickness": 2,
                                "strokeColor": None,
                                "colorRange": {
                                    "colors": [
                                        "#FF4040", "#F8671F", "#E49109", "#C5B900",
                                        "#9FDB05", "#75F317", "#4CFE34", "#29FC59",
                                        "#0FEC83", "#02D1AC", "#02ACD1", "#0F83EC",
                                        "#2959FC", "#4C34FE", "#7517F3", "#9F05DB",
                                        "#C500B9", "#E40991", "#F81F67", "#FF4040",
                                    ],
                                    "name": "Sinebow",
                                    "type": "cyclical",
                                    "category": "D3",
                                },
                                "strokeColorRange": {
                                    "name": "Global Warming",
                                    "type": "sequential",
                                    "category": "Uber",
                                    "colors": [
                                        "#4C0035", "#880030", "#B72F15",
                                        "#D6610A", "#EF9100", "#FFC300",
                                    ],
                                },
                                "radiusRange": [0, 50],
                                "filled": True,
                                "billboard": False,
                                "allowHover": True,
                                "showNeighborOnHover": False,
                                "showHighlightColor": True,
                            },
                            "hidden": False,
                            "textLabel": [
                                {
                                    "field": None,
                                    "color": [255, 255, 255],
                                    "size": 18,
                                    "offset": [0, 0],
                                    "anchor": "start",
                                    "alignment": "center",
                                    "outlineWidth": 0,
                                    "outlineColor": [255, 0, 0, 255],
                                    "background": False,
                                    "backgroundColor": [0, 0, 200, 255],
                                }
                            ],
                        },
                        "visualChannels": {
                            "colorField": {"name": "route_id", "type": "integer"},
                            "colorScale": "quantile",
                            "strokeColorField": None,
                            "strokeColorScale": "quantile",
                            "sizeField": None,
                            "sizeScale": "linear",
                        },
                    }
                ],
                "effects": [],
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": {
                            "CapMetro_Buses": [
                                {"name": "route_id", "format": None},
                                {"name": "trip_id", "format": None},
                                {"name": "vehicle_id", "format": None},
                                {"name": "speed", "format": None},
                            ]
                        },
                        "compareMode": False,
                        "compareType": "absolute",
                        "enabled": True,
                    },
                    "brush": {"size": 0.5, "enabled": False},
                    "geocoder": {"enabled": False},
                    "coordinate": {"enabled": False},
                },
                "layerBlending": "normal",
                "overlayBlending": "normal",
                "splitMaps": [],
                "animationConfig": {"currentTime": None, "speed": 1},
                "editor": {"features": [], "visible": True},
            },
            "mapState": {
                "bearing": 0,
                "dragRotate": False,
                "latitude": 30.27866105763493,
                "longitude": -97.7753201488513,
                "pitch": 0,
                "zoom": 10.16881985509215,
                "isSplit": False,
                "isViewportSynced": True,
                "isZoomLocked": False,
                "splitMapViewports": [],
            },
            "mapStyle": {
                "styleType": "dark",
                "topLayerGroups": {},
                "visibleLayerGroups": {
                    "label": False,
                    "road": True,
                    "border": False,
                    "building": True,
                    "water": True,
                    "land": True,
                    "3d building": False,
                },
                "threeDBuildingColor": [9.665468314072013, 17.18305478057247, 31.1442867897876],
                "backgroundColor": [0, 0, 0],
                "mapStyles": {},
            },
            "uiState": {"mapControls": {"mapLegend": {"active": False}}},
        },
    }

    austin_map = KeplerGl(
        height=600, data={"CapMetro_Buses": df}, config=map_config)
    file_name = "capmetro_map_wconfig.html"
    austin_map.save_to_html(file_name=file_name)
    webbrowser.open_new_tab("file://" + os.path.realpath(file_name))
    print(f"Kepler map saved to {file_name}")


def build_pydeck_map(df: pd.DataFrame) -> None:
    trips_data = df.groupby("trip_id").apply(
        lambda x: {
            "waypoints": x[["longitude", "latitude"]].values.tolist(),
            "timestamps": x["raw_timestamp"].tolist(),
            "route": str(x["route_id"].iloc[0]),
        },
        include_groups=False,
    ).tolist()

    layer = pdk.Layer(
        "TripsLayer",
        trips_data,
        get_path="waypoints",
        get_timestamps="timestamps",
        get_color=[255, 180, 0],  # Gold trails
        opacity=0.8,
        width_min_pixels=5,
        rounded=True,
        trail_length=600,  # 10 minutes of "tail"
        current_time=int(df["raw_timestamp"].min()),
    )

    view_state = pdk.ViewState(
        latitude=30.2672, longitude=-97.7431, zoom=12, pitch=40)

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )
    r.to_html("pydeck_austin_buses.html", open_browser=True)
    print("Pydeck map saved to pydeck_austin_buses.html")


def main():
    parser = argparse.ArgumentParser(description="Generate CapMetro bus maps.")
    parser.add_argument(
        "date", help="Date string in YYYY_MM_DD format (e.g. 2026_03_04)")
    parser.add_argument(
        "--map",
        choices=["kepler", "pydeck", "both"],
        default="both",
        help="Which map to generate (default: both)",
    )
    args = parser.parse_args()

    df = load_data(args.date)
    print(f"Loaded {len(df)} vehicle position records.")

    if args.map in ("kepler", "both"):
        build_kepler_map(df)
    if args.map in ("pydeck", "both"):
        build_pydeck_map(df)


if __name__ == "__main__":
    main()
