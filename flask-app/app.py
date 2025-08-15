import time
from flask import Flask, request, jsonify,render_template, Response,send_from_directory
from clustering_utils import process_customer_data_for_clustering_clust,get_closest_repzone_clust
from pyproj import Transformer # Import Transformer here if not already
import pandas as pd
import requests
import logging
import traceback
from dotenv import load_dotenv
import os
import json
import folium
import numpy as np
from sec_utils import (
    calculate_inertias,
    auto_select_k,
    balanced_kmeans_clustering,
    get_closest_repzone,
    is_too_far,
    prepare_weekly_schedule,
    assign_proximity,
    repzone_offices,compute_distance_matrices,solve_weekly_routes_far,solve_weekly_routes_near,
    summarize_routes, plot_cluster_routes
)
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

current_cluster = {"value": None}
load_dotenv()  # Loads from .env by default
#Get the end point of ors-app ,it is available in docker-compose.yml file in flask service
ORS_MATRIX_URL = os.getenv("ORS_API_URL")
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
ORS_MATRIX_URL = ORS_MATRIX_URL

@app.route('/print', methods=['GET'])
def print_check():
    print("I am here — /print endpoint was hit")
    return "Printed to server console", 200

@app.route('/test-distance', methods=['GET'])
def test_distance():
    import requests

    url = "http://host.docker.internal:8084/ors/v2/directions/driving-car"
    headers = {'Content-Type': 'application/json'}
    body = {
        "coordinates": [
            [32.8597, 39.9334],      # lon, lat (start)
            [32.8660, 39.9208]      # lon, lat (end)
        ]
    }

    try:
        response = requests.post(url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Optional: Print a quick summary
        summary = data['routes'][0]['summary']
        print(f"ORS returned: {summary['distance']} meters, {summary['duration']} seconds")

        return jsonify({
            "status": "success",
            "distance_m": summary['distance'],
            "duration_s": summary['duration']
        })

    except requests.exceptions.RequestException as e:
        print(f"ORS request failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/ors-health', methods=['GET'])
def ors_health_check():
    try:
        response = requests.get("http://host.docker.internal:8084/ors/v2/health", timeout=5)
        if response.status_code == 200:
            return jsonify({"status": "healthy", "response": response.json()})
        else:
            return jsonify({"status": "unhealthy", "code": response.status_code, "response": response.text}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/solve')
def solve():
    # Distance callback
    def create_distance_callback():
        distances = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0]
        ]
        return lambda from_node, to_node: distances[from_node][to_node]

    # Create Routing Index Manager and Routing Model
    manager = pywrapcp.RoutingIndexManager(4, 1, 0)  # 4 locations, 1 vehicle, start at index 0
    routing = pywrapcp.RoutingModel(manager)

    distance_callback = create_distance_callback()
    transit_callback_index = routing.RegisterTransitCallback(
        lambda from_index, to_index: distance_callback(manager.IndexToNode(from_index), manager.IndexToNode(to_index))
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Extract the route
    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        return jsonify({"route": route})
    else:
        return jsonify({"error": "No solution found"})

@app.route('/clust', methods=['GET', 'POST'])
def perform_clustering_and_distance_matrices():
    start_time = time.time()
    log_messages = []

    # --- Load Customer Table from POST JSON or fallback to static file ---
    if request.is_json:
        json_data = request.get_json()
        if 'customer_table' in json_data:
            df_Customer_Table = pd.DataFrame(json_data['customer_table'])
        else:
            df_Customer_Table = pd.read_json("static/customer_table.json")
    else:
        df_Customer_Table = pd.read_json("static/customer_table.json")

    # --- Project to UTM ---
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    utm_coords = np.array([
        transformer.transform(lon, lat) for lat, lon in zip(df_Customer_Table['Latitude'], df_Customer_Table['Longitude'])
    ])

    # --- Initial Clustering ---
    inertias = calculate_inertias(utm_coords)
    best_k = auto_select_k(inertias, (5, 30))
    
    df_balanced = balanced_kmeans_clustering(df_Customer_Table, initial_k=best_k)

    # --- RepZone Assignment ---
    df_centroids = df_balanced.groupby('cluster').agg({'Latitude': 'mean', 'Longitude': 'mean'}).reset_index()
    df_centroids['Assigned_RepZone'] = df_centroids.apply(
        lambda row: get_closest_repzone(row['Latitude'], row['Longitude']), axis=1
    )
    df_balanced = df_balanced.merge(df_centroids[['cluster', 'Assigned_RepZone']], on='cluster')

    repzone_coords = [(lon, lat) for lat, lon in repzone_offices.values()]
    df_balanced = df_balanced[df_balanced.apply(lambda row: not is_too_far(row, repzone_coords), axis=1)]

    # --- Reproject to EPSG:5254 and Recluster ---
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5254", always_xy=True)
    utm_coords = np.array([
        transformer.transform(lon, lat) for lat, lon in zip(df_balanced['Latitude'], df_balanced['Longitude'])
    ])
    inertias = calculate_inertias(utm_coords)
    best_k = auto_select_k(inertias, (5, 30))
    df_balanced = balanced_kmeans_clustering(df_balanced, initial_k=best_k)

    # --- Final Centroid + RepZone Assignment ---
    df_centroids = df_balanced.groupby('cluster').agg({'Latitude': 'mean', 'Longitude': 'mean'}).reset_index()
    df_centroids['Assigned_RepZone'] = df_centroids.apply(
        lambda row: get_closest_repzone(row['Latitude'], row['Longitude']), axis=1
    )
    df_balanced = df_balanced.merge(df_centroids[['cluster', 'Assigned_RepZone']], on='cluster')

    # --- Weekly Schedule + Distance Matrices ---
    df_schedule = prepare_weekly_schedule(df_balanced)
    dur_mats, dist_mats, id_maps = compute_distance_matrices(df_schedule, df_centroids)
    df_centroids.to_json("static/df_centroids.json", orient="records", indent=2)
    df_schedule.to_json("static/df_schedule.json", orient="records", indent=2)

    # --- Routing Logic ---
  
    # cluster_list = [1,2,5]
    cluster_list=df_schedule['cluster'].unique().tolist()

    week = 2
    routes_all = {}
    summaries_all = {}
    far_dict={}

    for cluster in cluster_list:
        log_messages.append(f"Processing cluster {cluster}")
        current_cluster["value"] = f"Processing cluster {cluster}"
        real_durations = dur_mats[(cluster, week)]
        real_distances = dist_mats[(cluster, week)]
        customer_ids = id_maps[(cluster, week)]
        num_vehicles = 80

        # Test proximity

        df_centroids = assign_proximity(df_centroids, repzone_offices, threshold_km=300)
        
        centroid_string_distance=df_centroids[df_centroids['cluster']==cluster]['RepZone_Proximity'].item()
        
        # --- Call different function depending if cluster is near or far --- 

        if centroid_string_distance=='near':
    
            routes = solve_weekly_routes_near(real_durations,customer_ids, num_vehicles, max_work_time=480)

        else:

            num_vehicles = 20
            routes = solve_weekly_routes_far(real_durations,customer_ids, num_vehicles,max_work_time=800)



        # routes_standard = solve_weekly_routes_standard(real_durations, customer_ids, num_vehicles, max_work_time=480)

        # Just to test the connection

        # routes_standard={}

        if not routes:
            routes_all[f'cluster_{cluster}_week_{week}'] = 'No valid route found'
            log_messages.append(f"No valid route found for cluster {cluster}")
            continue

        summary_df = summarize_routes(
            routes, customer_ids,
            real_durations=real_durations,
            real_distances=real_distances,
            service_time=30
        )

        summary_json = summary_df.sort_values("total_time_min", ascending=False).to_dict(orient='records')
        routes_all[f'cluster_{cluster}_week_{week}'] = routes
        summaries_all[f'cluster_{cluster}_week_{week}'] = summary_json

    def convert_dict(d):
        return {
            str(k): (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in d.items()
        }


    result = {
        "dur_mats": convert_dict(dur_mats),
        "dist_mats": convert_dict(dist_mats),
        "id_maps": convert_dict(id_maps),
        "routes": routes_all,
        "summaries": summaries_all
    }

    

    end_time = time.time()
    result["runtime_seconds"] = round(end_time - start_time, 2)
    result["log"] = log_messages

    # Export to a fixed filename
    with open("static/results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return jsonify(result)


@app.route('/clust-progress')
def clust_progress_stream():
    def event_stream():
        last_sent = None
        while True:
            if current_cluster["value"] != last_sent:
                yield f"data: {current_cluster['value']}\n\n"
                last_sent = current_cluster["value"]
            time.sleep(0.5)
    return Response(event_stream(), mimetype='text/event-stream')








@app.route('/visual', methods=['GET', 'POST'])
def index():
    customer_locations = []
    office_locations = []
    map_html = ""
    assignment_map_html = ""  # Will be unused in this version

    with open("static/office_locations.json", "r") as file:
        office_data = json.load(file)

    if request.method == 'POST':
        customer_file = request.files.get('customer_file')
        min_cluster_size = int(request.form.get('min_cluster_size') or 30)
        max_cluster_size = int(request.form.get('max_cluster_size') or 300)
        enable_clustering = 'enable_clustering' in request.form

        try:
            customer_data = []
            if customer_file:
                customer_data = json.load(customer_file)
                if isinstance(customer_data, list):
                    customer_locations = [
                        (float(item.get('Latitude')), float(item.get('Longitude')), item.get("CustomerId"))
                        for item in customer_data
                        if isinstance(item, dict) and 'Latitude' in item and 'Longitude' in item
                    ]
                else:
                    return render_template('index.html', error="Invalid customer data format.")
            
            office_locations = [
                (float(item.get('Latitude')), float(item.get('Longitude')), item.get("city"))
                for item in office_data
                if isinstance(item, dict) and 'Latitude' in item and 'Longitude' in item
            ]

            all_coords = [coord[:2] for coord in customer_locations + office_locations if coord[:2] != (None, None)]
            avg_lat, avg_lon = (sum(c[0] for c in all_coords) / len(all_coords), 
                                sum(c[1] for c in all_coords) / len(all_coords)) if all_coords else (39.9255, 32.8663)

            # Initialize single map
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)

            # Layers
            rep_layer = folium.FeatureGroup(name='Repzone Representatives')
            customer_layer = folium.FeatureGroup(name='Repzone Customers')
            clustered_layer = folium.FeatureGroup(name='Clustered Customers')
            centroids_layer = folium.FeatureGroup(name='Cluster Centroids w/ Assigned RepZone')
            assigned_customers_layer = folium.FeatureGroup(name='Customers Colored by Assigned RepZone')

            for lat, lon, cust_id in customer_locations:
                folium.Marker(
                    [lat, lon],
                    popup=f"Customer Id: {cust_id}",
                    icon=folium.Icon(color='blue', icon='user')
                ).add_to(customer_layer)

            for lat, lon, city in office_locations:
                folium.Marker(
                    [lat, lon],
                    popup=f"Repzone office at {city}",
                    icon=folium.Icon(color='red', icon='home')
                ).add_to(rep_layer)

            if enable_clustering and customer_data:
                clustered_data, cluster_centers = process_customer_data_for_clustering_clust(
                    customer_data,
                    min_size=min_cluster_size,
                    max_size=max_cluster_size
                )

                unique_clusters = set(item['cluster'] for item in clustered_data)
                base_colors = ['green', 'purple', 'orange', 'darkblue', 'darkred', 'brown',
                               'lightred', 'beige', 'darkgreen', 'darkpurple', 'cadetblue',
                               'pink', 'lightblue', 'lightgreen']
                while len(base_colors) < len(unique_clusters):
                    base_colors *= 2
                cluster_color_map = {cluster: base_colors[i] for i, cluster in enumerate(unique_clusters)}

                for item in clustered_data:
                    cluster_id = item['cluster']
                    folium.CircleMarker(
                        [float(item['Latitude']), float(item['Longitude'])],
                        radius=8,
                        popup=f"Customer: {item.get('CustomerId', 'N/A')}<br>Cluster: {cluster_id}",
                        color=cluster_color_map.get(cluster_id, 'darkred'),
                        fill=True,
                        fill_color=cluster_color_map.get(cluster_id, 'darkred'),
                        fill_opacity=0.7
                    ).add_to(clustered_layer)

                df_centroids = pd.DataFrame([
                    {"cluster": cid, "Latitude": info['lat'], "Longitude": info['lon']}
                    for cid, info in cluster_centers.items()
                ])
                df_centroids['Assigned_RepZone'] = df_centroids.apply(
                    lambda row: get_closest_repzone_clust(row['Latitude'], row['Longitude']), axis=1
                )
                cluster_to_repzone = df_centroids.set_index('cluster')['Assigned_RepZone'].to_dict()
                repzone_names = df_centroids['Assigned_RepZone'].unique()
                repzone_colors = base_colors.copy()
                while len(repzone_colors) < len(repzone_names):
                    repzone_colors *= 2
                repzone_color_map = {name: repzone_colors[i] for i, name in enumerate(repzone_names)}

                for _, row in df_centroids.iterrows():
                    color = repzone_color_map.get(row['Assigned_RepZone'], 'darkred')
                    folium.Marker(
                        [row['Latitude'], row['Longitude']],
                        popup=f"Cluster {row['cluster']}<br>Assigned RepZone: {row['Assigned_RepZone']}",
                        icon=folium.Icon(color=color, icon='flag')
                    ).add_to(centroids_layer)

                for item in clustered_data:
                    cluster_id = item['cluster']
                    repzone = cluster_to_repzone.get(cluster_id)
                    color = repzone_color_map.get(repzone, 'darkred')
                    folium.CircleMarker(
                        [float(item['Latitude']), float(item['Longitude'])],
                        radius=6,
                        popup=f"Customer: {item.get('CustomerId', 'N/A')}<br>Cluster: {cluster_id}<br>Assigned RepZone: {repzone}",
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.6
                    ).add_to(assigned_customers_layer)

            # Add all layers
            rep_layer.add_to(m)
            customer_layer.add_to(m)
            if enable_clustering and customer_data:
                clustered_layer.add_to(m)
                centroids_layer.add_to(m)
                assigned_customers_layer.add_to(m)

            folium.LayerControl(collapsed=False).add_to(m)
            map_html = m.get_root().render()

        except json.JSONDecodeError:
            return render_template('index.html', error="Error decoding JSON.")
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {e}")

    if not map_html:
        map_html = "<p>No map generated.</p>"

    return render_template("index.html", map_html=map_html)  # assignment_map_html unused



@app.route('/generate_map', methods=['GET'])
def generate_map():
    # Load core data
    with open("static/results.json", "r", encoding="utf-8") as f:
        result = json.load(f)

    routes_all = result["routes"]
    summaries_all = result["summaries"]

    # Load DataFrames
    df_schedule = pd.read_json("static/df_schedule.json")
    df_centroids = pd.read_json("static/df_centroids.json")

    # ---- Rebuild df_cluster_summary ----
    records = []
    for key, summary_list in summaries_all.items():
        parts = key.split('_')
        cluster = parts[1]
        week = parts[3]

        total_distance = sum(vehicle['total_distance_km'] for vehicle in summary_list)
        num_vehicles = len(summary_list)

        records.append({
            "cluster": int(cluster),
            "week": int(week),
            "total_vehicles": num_vehicles,
            "total_distance_km": total_distance
        })

    df_cluster_summary = pd.DataFrame(records).sort_values(by=["week", "cluster"]).reset_index(drop=True)
    df_centroids["cluster"] = df_centroids["cluster"].astype(int)
    df_cluster_summary["cluster"] = df_cluster_summary["cluster"].astype(int)

    df_cluster_summary = pd.merge(
        df_cluster_summary,
        df_centroids[["cluster", "Assigned_RepZone"]],
        on="cluster",
        how="left"
    )

    # ---- Generate map for each cluster ----
    success = []
    failures = []

    for cluster_key in routes_all.keys():
        try:
            m = plot_cluster_routes(cluster_key, routes_all, df_schedule, df_cluster_summary, repzone_offices)
            map_path = f"static/maps/{cluster_key}.html"
            m.save(map_path)
            success.append(cluster_key)
        except Exception as e:
            print(f"Failed to render {cluster_key}: {e}")
            failures.append({"cluster": cluster_key, "error": str(e)})

    return jsonify({
        "status": "completed",
        "rendered_maps": success,
        "failed": failures
    })

@app.route('/list_maps')
def list_maps():
    try:
        map_files = [
            f for f in os.listdir("static/maps") 
            if f.endswith(".html")
        ]
        return jsonify(map_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory('static/maps', filename)

@app.route('/client_info/<client_id>', methods=['GET'])
def get_client_info(client_id):
    try:
        client_id = int(client_id)
    except ValueError:
        return jsonify({"error": f"Invalid client ID format: {client_id}"}), 400

    # Load data
    try:
        with open("static/results.json", "r", encoding="utf-8") as f:
            result = json.load(f)

        routes_all = result["routes"]
        summaries_all = result["summaries"]
        df_schedule = pd.read_json("static/df_schedule.json")
        df_centroids = pd.read_json("static/df_centroids.json")

        # Build df_cluster_summary
        records = []
        for key, summary_list in summaries_all.items():
            parts = key.split('_')
            cluster = int(parts[1])
            week = int(parts[3])
            total_distance = sum(vehicle['total_distance_km'] for vehicle in summary_list)
            num_vehicles = len(summary_list)
            records.append({
                "cluster": cluster,
                "week": week,
                "total_vehicles": num_vehicles,
                "total_distance_km": total_distance
            })

        df_cluster_summary = pd.DataFrame(records).sort_values(by=["week", "cluster"]).reset_index(drop=True)
        df_centroids["cluster"] = df_centroids["cluster"].astype(int)
        df_cluster_summary["cluster"] = df_cluster_summary["cluster"].astype(int)

        df_cluster_summary = pd.merge(
            df_cluster_summary,
            df_centroids[["cluster", "Assigned_RepZone"]],
            on="cluster",
            how="left"
        )

    except Exception as e:
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

    # Search for client_id in routes
    found = False
    for cluster_key, vehicle_routes in routes_all.items():
        for vehicle_id, route in vehicle_routes.items():
            if client_id in route:
                found = True
                break
        if found:
            break

    if not found:
        return jsonify({"error": f"Client ID {client_id} not found in any route."}), 404

    # Extract cluster and week from key
    parts = cluster_key.split('_')
    cluster = int(parts[1])
    week = int(parts[3])

    assigned_row = df_cluster_summary[
        (df_cluster_summary["cluster"] == cluster) &
        (df_cluster_summary["week"] == week)
    ]

    repzone = "Unknown"
    if not assigned_row.empty:
        repzone = assigned_row["Assigned_RepZone"].values[0]

    vehicle_route = vehicle_routes[vehicle_id]

    return jsonify({
        "client_id": client_id,
        "repzone_office": repzone,
        "cluster": cluster,
        "week": week,
        "vehicle_route": vehicle_route
    })

@app.route('/')
def clustering_ui():
    return render_template('clustering_ui.html')


if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)
