import pandas as pd
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from sklearn.cluster import KMeans
from pyproj import Proj, Transformer

from kneed import KneeLocator
from haversine import haversine, Unit



from sklearn.cluster import KMeans
from pyproj import Transformer
import numpy as np
import pandas as pd

import requests
from time import sleep
from dotenv import load_dotenv
import os

from geopy.distance import geodesic

import folium
from matplotlib.cm import get_cmap
import polyline
from folium import Marker, CircleMarker, PolyLine, Icon, LayerControl


load_dotenv()  # Loads from .env by default
#Get the end point of ors-app ,it is available in docker-compose.yml file in flask service
ORS_MATRIX_URL = os.getenv("ORS_API_URL")

def get_ors_duration_distance(origins, destinations):
    '''
    Calculate the distance and travel duration matrices, 
    using local calculation on Open Route Service.

    '''
    locations = [[lon, lat] for lat, lon in origins + destinations]
    body = {
        "locations": locations,
        "metrics": ["duration", "distance"],
        "sources": list(range(len(origins))),
        "destinations": list(range(len(origins), len(destinations) + len(origins))),
    }

    response = requests.post(ORS_MATRIX_URL, json=body)

    if response.status_code == 200:
        data = response.json()
        durations = np.array(data["durations"]) / 60
        distances = np.array(data["distances"]) / 1000
        return durations.astype(int), distances
    else:
        print("ORS request failed:", response.text)
        return None, None


def compute_distance_matrices_for_cluster(df_schedule, cluster_id, RepZone, batch_size=50, sleep_time=0.5):
    """
    Computes distance and duration matrices for one cluster over 4 weeks.
    Includes RepZone as the origin depot (index 0).

    Input:
    ------

    df_schedule: DataFrame with geographic cluster and weekly schedule assigned.

    cluster_id: Cluster to be considered.

    RepZone: Origin coordinates in the format (lon, lat)

    batch_size: Default 50.

    sleep_time: Default 0.5



    Output:
    -------
    
    duration_matrices: {week: np.array}
    
    distance_matrices: {week: np.array}
    
    index_to_customer_id: {week: list}
    """
    duration_matrices = {}
    distance_matrices = {}
    index_to_customer_id = {}

    cluster_df = df_schedule[df_schedule['cluster'] == cluster_id]

    for week in sorted(cluster_df['week'].unique()):
        week_df = cluster_df[cluster_df['week'] == week]

        client_coords = week_df[['Latitude', 'Longitude']].values
        repzone_coords = (RepZone[1], RepZone[0])  # lon, lat → lat, lon
        all_coords = np.vstack([repzone_coords, client_coords])
        n = len(all_coords)

        # Map index to CustomerId (0 is RepZone)
        ids = ['RepZone'] + week_df['CustomerId'].tolist()
        index_to_customer_id[week] = ids

        # Init matrices
        durations = np.full((n, n), 99999, dtype=int)
        distances = np.full((n, n), 9999, dtype=float)

        for i in range(0, n, batch_size):
            batch_origins = all_coords[i:i + batch_size]

            for j in range(0, n, batch_size):
                batch_destinations = all_coords[j:j + batch_size]

                dur, dist = get_ors_duration_distance(batch_origins.tolist(), batch_destinations.tolist())
                if dur is not None and dist is not None:
                    durations[i:i + batch_size, j:j + batch_size] = dur
                    distances[i:i + batch_size, j:j + batch_size] = dist

            sleep(sleep_time)

        np.fill_diagonal(durations, 180)
        np.fill_diagonal(distances, 0)


        duration_matrices[week] = durations
        distance_matrices[week] = distances

        print(f"Cluster {cluster_id} Week {week}: {n-1} clients")

    return duration_matrices, distance_matrices, index_to_customer_id


repzone_offices = {
    "Istanbul": [40.9923, 29.0800],
    "Adana": [37.0015, 35.3213],
    "Ankara": [39.9334, 32.8597],
    "Izmir": [38.4707, 27.1417],
    "Tekirdağ": [41.1872, 27.8833]
}
def auto_select_k(inertias, k_range):
    ks = list(range(*k_range))
    kl = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
    return kl.elbow

def calculate_inertias(utm_coords, k_range=(5, 30)):
    inertias = []
    ks = range(*k_range)
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(utm_coords)
        inertias.append(kmeans.inertia_)
    return inertias
def prepare_weekly_schedule(df, num_weeks=4, random_state=42):
    """
    For each cluster, distribute weekly and monthly clients across a 4-week horizon.
    Weekly clients appear in all weeks; monthly clients are evenly split.

    Input:
    ------
    df: DataFrame
    num_weeks: The future horizon planning, default is 4.
    
    
    Output:
    -------
    
    Returns a new DataFrame with a `week` column.

    
    """
    df = df.copy()
    all_weeks = []

    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id]

        # Weekly = 1, Monthly = 4
        weekly_clients = cluster_df[cluster_df['VisitFreq'] == 1]
        monthly_clients = cluster_df[cluster_df['VisitFreq'] == 4]

        # Shuffle monthly clients and split them into chunks for each week
        monthly_split = np.array_split(
            monthly_clients.sample(frac=1, random_state=random_state),
            num_weeks
        )

        for week in range(1, num_weeks + 1):
            weekly_part = weekly_clients.copy()
            weekly_part['week'] = week

            monthly_part = monthly_split[week - 1].copy()
            monthly_part['week'] = week

            combined = pd.concat([weekly_part, monthly_part], ignore_index=True)
            combined['cluster'] = cluster_id
            all_weeks.append(combined)

    return pd.concat(all_weeks).reset_index(drop=True)

def get_closest_repzone(client_lat, client_lon):
    distances = {}
    for office, coords in repzone_offices.items():
        dist = geodesic((client_lat, client_lon), tuple(coords)).km  # distance in km
        distances[office] = dist
    closest_office = min(distances, key=distances.get)
    return closest_office
def is_too_far(row, repzones, threshold_km=500):
    client_coord = (row['Longitude'], row['Latitude'])
    distances = [haversine(client_coord, office, unit=Unit.KILOMETERS) for office in repzones]
    return all(d > threshold_km for d in distances)

from geopy.distance import geodesic
def balanced_kmeans_clustering(
    df,
    lat_col='Latitude',
    lon_col='Longitude',
    min_size=30,
    max_size=300,
    initial_k=10,
    random_state=42,
):
    '''
    Function to calculate clusters given parameters to determine maximum and minimum sizes, using KNN.

    Input:
    -----
    df: DataFrame with geographic data to cluster.
    
    lat_col: Key for latitude, default is 'Latitude'
    
    lon_col: Key for longitude, default is 'Longitude'
    
    min_size: Default size is 30.
    
    max_size: Default size is 300.
    
    initial_k: Default is 10.

    Output:
    ------

    df with cluster column
    '''
    
    # Step 1: Project coordinates to UTM coordinates
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5254", always_xy=True)
    utm_coords = np.array([
        transformer.transform(lon, lat) for lat, lon in zip(df[lat_col], df[lon_col])
    ])
    df = df.copy()
    df['x'] = utm_coords[:, 0]
    df['y'] = utm_coords[:, 1]

    # Step 2: Initial KMeans, assuming initial_k number of clusters
    
    kmeans = KMeans(n_clusters=initial_k, random_state=random_state).fit(utm_coords)
    df['cluster'] = kmeans.labels_

    # Step 3: Refine clusters to enforce min/max size
    next_cluster_id = df['cluster'].max() + 1
    changed = True
    while changed:
        changed = False
        cluster_sizes = df['cluster'].value_counts().to_dict()

        for cluster_id, size in cluster_sizes.items():
            subset = df[df['cluster'] == cluster_id]
            if size > max_size:
                # Split large cluster into subclusters
                n_subclusters = int(np.ceil(size / max_size))
                sub_kmeans = KMeans(n_clusters=n_subclusters, random_state=random_state)
                sub_labels = sub_kmeans.fit_predict(subset[['x', 'y']])
                for i, new_label in enumerate(sub_labels):
                    df.loc[subset.index[i], 'cluster'] = next_cluster_id + new_label
                next_cluster_id += n_subclusters
                changed = True

            elif size < min_size:
                # Merge small cluster to closest centroid
                subset_coords = subset[['x', 'y']].mean().values
                # Compute distances to other centroids
                other_clusters = df['cluster'].unique()
                dists = []
                for other_id in other_clusters:
                    if other_id == cluster_id:
                        continue
                    other_coords = df[df['cluster'] == other_id][['x', 'y']].mean().values
                    dist = np.linalg.norm(subset_coords - other_coords)
                    dists.append((other_id, dist))
                if dists:
                    closest_id = min(dists, key=lambda x: x[1])[0]
                    df.loc[df['cluster'] == cluster_id, 'cluster'] = closest_id
                    changed = True

    return df.drop(columns=['x', 'y'])
def get_closest_repzone(client_lat, client_lon):
    distances = {}
    for office, coords in repzone_offices.items():
        dist = geodesic((client_lat, client_lon), tuple(coords)).km  # distance in km
        distances[office] = dist
    closest_office = min(distances, key=distances.get)
    return closest_office
def compute_distance_matrices(df_schedule,df_centroids, batch_size=50, sleep_time=0.5):
    """
    Computes distance and duration matrices for one cluster over 4 weeks.
    Includes RepZone as the origin depot (index 0).

    Input:
    ------

    df_schedule: DataFrame with geographic cluster and weekly schedule assigned.

    batch_size: Default 50.

    sleep_time: Default 0.5



    Output:
    -------
    
    duration_matrices: {(cluster_id,week): np.array}
    
    distance_matrices: {(cluster_id,week): np.array}
    
    index_to_customer_id: {(cluster_id,week): list}
    """
    
    duration_matrices = {}
    distance_matrices = {}
    index_to_customer_id = {}

    # cluster_lists=[2,7]

    cluster_lists=df_schedule['cluster'].unique().tolist()
    
    for cluster_id in cluster_lists:

        cluster_df = df_schedule[df_schedule['cluster'] == cluster_id]

        # Find out the office assigned to the cluster we want to analyze
        RepZone_test=df_centroids[df_centroids['cluster']==cluster_id]['Assigned_RepZone'].tolist()
        
        # Find the (Lat, Lon) coordinates of the office
        RepZone_test_Lat_Lon=repzone_offices[RepZone_test[0]]
        
        # Invert to  (Lon, Lat)
        RepZone = (RepZone_test_Lat_Lon[1], RepZone_test_Lat_Lon[0])   
    
        for week in sorted(cluster_df['week'].unique()):
            week_df = cluster_df[cluster_df['week'] == week]
    
            client_coords = week_df[['Latitude', 'Longitude']].values
            repzone_coords = (RepZone[1], RepZone[0])  # lon, lat → lat, lon
            all_coords = np.vstack([repzone_coords, client_coords])
            n = len(all_coords)
    
            # Map index to CustomerId (0 is RepZone)
            ids = ['RepZone'] + week_df['CustomerId'].tolist()
            index_to_customer_id[(cluster_id,week)] = ids
    
            # Init matrices
            durations = np.full((n, n), 99999, dtype=int)
            distances = np.full((n, n), 9999, dtype=float)
    
            for i in range(0, n, batch_size):
                batch_origins = all_coords[i:i + batch_size]
    
                for j in range(0, n, batch_size):
                    batch_destinations = all_coords[j:j + batch_size]
    
                    dur, dist = get_ors_duration_distance(batch_origins.tolist(), batch_destinations.tolist())
                    if dur is not None and dist is not None:
                        durations[i:i + batch_size, j:j + batch_size] = dur
                        distances[i:i + batch_size, j:j + batch_size] = dist
    
                sleep(sleep_time)
    
            np.fill_diagonal(durations, 2000)
            np.fill_diagonal(distances, 0)
    
            duration_matrices[(cluster_id,week)] = durations
            distance_matrices[(cluster_id,week)] = distances
    
            print(f"Cluster {cluster_id} Week {week}: {n-1} clients")

    return duration_matrices, distance_matrices, index_to_customer_id
def normalize_key(k):
    # Remove unwanted text
    if isinstance(k, str):
        k = k.replace("np.int64(", "").replace(")", "").replace("'", "")
        parts = k.strip("()").split(", ")
        return f"cluster_{parts[0]}_week_{parts[1]}"
    return k
def fetch_ors_directions(locations, profile="driving-car"):
    url = f"http://host.docker.internal:8084/ors/v2/directions/{profile}"
    headers = {"Content-Type": "application/json"}
    coords = [[lon, lat] for lat, lon in locations]

    body = {
        "coordinates": coords,
        "format": "geojson"
    }

    try:
        response = requests.post(url, json=body, headers=headers)
        data = response.json()

        if 'features' in data and 'geometry' in data['features'][0]:
            road_coords = data["features"][0]["geometry"]["coordinates"]
            return [(lat, lon) for lon, lat in road_coords]

        elif 'routes' in data and 'geometry' in data['routes'][0]:
            encoded = data['routes'][0]['geometry']
            return polyline.decode(encoded)

        else:
            print("Unrecognized ORS format:", data)
            return None

    except Exception as e:
        print("ORS parsing failed:", e)
        return None

def summarize_routes(routes, customer_ids, real_durations, real_distances, service_time=30):
    '''
    Computes total time and distance for each route and returns a summary DataFrame.

    Input:
    ------

    routes: The output of OR-tools, designated routes.

    real_durations: Matrix with travel times calculated with Open Route Service.

    real_distances: Matrix with travel distances calculated with Open Route Service.

    service_time: Service time at each stop, default is 30 minutes.
    
    
    Output:
    -------
    
    DataFrame with columns: vehicle_id, total_time_min, total_distance_km, num_stops
        
    '''
    
    id_to_index = {cid: idx for idx, cid in enumerate(customer_ids)}
    records = []

    for vehicle_id, route_ids in routes.items():
        # Time
        total_time = 0
        for i in range(len(route_ids) - 1):
            from_idx = id_to_index[route_ids[i]]
            to_idx = id_to_index[route_ids[i + 1]]
            travel = real_durations[from_idx][to_idx]
            stop = 0 if route_ids[i + 1] == "RepZone" else service_time
            total_time += travel + stop

        # Distance
        total_distance = 0.0
        for i in range(len(route_ids) - 1):
            from_idx = id_to_index[route_ids[i]]
            to_idx = id_to_index[route_ids[i + 1]]
            dist = real_distances[from_idx][to_idx]
            total_distance += dist

        # Store summary
        records.append({
            "vehicle_id": vehicle_id,
            "total_time_min": total_time,
            "total_distance_km": total_distance,
            "num_stops": len(route_ids) - 1  # excluding depot
        })

    return pd.DataFrame(records)


def solve_weekly_routes_standard(real_durations,customer_ids, num_vehicles, SERVICE_TIME=30, estimated_number_vehicles=7, max_work_time=480):
    '''
    Solves the routing problem for one cluster-week using OR-Tools.

    Input:
    ------
    
    real_durations (np.array): Duration matrix including RepZone at index 0.

    real_distances (np.array): Distance matrix including RepZone at index 0. (km)
    
    customer_ids (list): Original customer IDs, where customer_ids[0] == 'RepZone'.
    
    num_vehicles (int): Number of reps * days for this schedule.
    
    SERVICE_TIME (int): Service time at each stop in minutes.
    
    max_work_time (int): Max total work time per vehicle (in minutes).

    Output:
    -------
    
    dict: vehicle_id → list of CustomerIds (route)
    
    '''


    data = {
        'time_matrix': real_durations.tolist(),
        'num_vehicles': num_vehicles,
        'depot': 0
    }

    manager = pywrapcp.RoutingIndexManager(
        len(data['time_matrix']),
        data['num_vehicles'],
        [data['depot']] * data['num_vehicles'],
        [data['depot']] * data['num_vehicles']
    )

    routing = pywrapcp.RoutingModel(manager)

    def total_time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data['time_matrix'][from_node][to_node]
        service_time = 0 if from_node == 0 else SERVICE_TIME
        return travel_time + service_time

    total_time_callback_index = routing.RegisterTransitCallback(total_time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(total_time_callback_index)

 
    # Add time dimension
    routing.AddDimension(
        total_time_callback_index,
        1200,              # slack
        10000,     # hard upper bound, very large
        True,              # start cumul at zero
        "Time"
    )

    time_dimension = routing.GetDimensionOrDie("Time")
    # time_dimension.SetGlobalSpanCostCoefficient(1000)

    


 
    

    for vehicle_id in range(num_vehicles):
        end_index = routing.End(vehicle_id)
        start_index = routing.Start(vehicle_id)

        time_dimension.SetCumulVarSoftUpperBound(end_index, max_work_time, 100)
        

        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(end_index))
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(start_index))

     

    # Hard constraint: each customer visited once
    penalty = 10000
    for customer_idx in range(1, len(data['time_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(customer_idx)], penalty)

    # Loosened VisitCount dimension
    visit_callback_index = routing.RegisterUnaryTransitCallback(lambda index: 1)
    routing.AddDimension(
        visit_callback_index,
        0,                  # no slack
        30,                 # allow many visits per rep, just soft-balance
        True,
        "VisitCount"
    )
    visit_dimension = routing.GetDimensionOrDie("VisitCount")
    for vehicle_id in range(num_vehicles):
        end_index = routing.End(vehicle_id)
        # Encourage soft upper limit of 9 visits per rep
        visit_dimension.SetCumulVarSoftUpperBound(end_index, 9, 100)


    # Search
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 100

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        print("No feasible solution found.")
        return None

    routes = {}
    new_vehicle_id = 0

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append('RepZone' if node == 0 else customer_ids[node])
            index = solution.Value(routing.NextVar(index))

        if len(route) > 1:
            routes[new_vehicle_id] = route
            new_vehicle_id += 1

    return routes
def solve_weekly_routes_near(real_durations,customer_ids, num_vehicles, SERVICE_TIME=30, max_work_time=480):
    '''
    Solves the routing problem for one cluster-week using OR-Tools.

    Input:
    ------
    
    real_durations (np.array): Duration matrix including RepZone at index 0.

    real_distances (np.array): Distance matrix including RepZone at index 0. (km)
    
    customer_ids (list): Original customer IDs, where customer_ids[0] == 'RepZone'.
    
    num_vehicles (int): Number of reps * days for this schedule.
    
    SERVICE_TIME (int): Service time at each stop in minutes.
    
    max_work_time (int): Max total work time per vehicle (in minutes).

    Output:
    -------
    
    dict: vehicle_id → list of CustomerIds (route)
    
    '''


    data = {
        'time_matrix': real_durations.tolist(),
        'num_vehicles': num_vehicles,
        'depot': 0
    }

    manager = pywrapcp.RoutingIndexManager(
        len(data['time_matrix']),
        data['num_vehicles'],
        [data['depot']] * data['num_vehicles'],
        [data['depot']] * data['num_vehicles']
    )

    routing = pywrapcp.RoutingModel(manager)

    def total_time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data['time_matrix'][from_node][to_node]
        service_time = 0 if from_node == 0 else SERVICE_TIME
        return travel_time + service_time

    total_time_callback_index = routing.RegisterTransitCallback(total_time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(total_time_callback_index)

 
    # Add time dimension
    routing.AddDimension(
        total_time_callback_index,
        1200,              # slack
        10000,     # hard upper bound, very large
        True,              # start cumul at zero
        "Time"
    )

    time_dimension = routing.GetDimensionOrDie("Time")
    # time_dimension.SetGlobalSpanCostCoefficient(1000)

    


 
    

    for vehicle_id in range(num_vehicles):
        end_index = routing.End(vehicle_id)
        start_index = routing.Start(vehicle_id)

        time_dimension.SetCumulVarSoftUpperBound(end_index, max_work_time, 100)
        

        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(end_index))
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(start_index))

     

    # Hard constraint: each customer visited once
    penalty = 10000
    for customer_idx in range(1, len(data['time_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(customer_idx)], penalty)

    # Loosened VisitCount dimension
    visit_callback_index = routing.RegisterUnaryTransitCallback(lambda index: 1)
    routing.AddDimension(
        visit_callback_index,
        0,                  # no slack
        30,                 # allow many visits per rep, just soft-balance
        True,
        "VisitCount"
    )
    visit_dimension = routing.GetDimensionOrDie("VisitCount")
    for vehicle_id in range(num_vehicles):
        end_index = routing.End(vehicle_id)
        # Encourage soft upper limit of 9 visits per rep
        visit_dimension.SetCumulVarSoftUpperBound(end_index, 9, 100)


    # Search
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 100

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        print("No feasible solution found.")
        return None

    routes = {}
    new_vehicle_id = 0

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append('RepZone' if node == 0 else customer_ids[node])
            index = solution.Value(routing.NextVar(index))

        if len(route) > 1:
            routes[new_vehicle_id] = route
            new_vehicle_id += 1

    return routes

def assign_proximity(df, repzone_offices_adresses, threshold_km=300):
    """
    Adds columns to the DataFrame:
    - 'Distance_to_RepZone_km': Geodesic distance from centroid to assigned office.
    - 'RepZone_Proximity': 'near' if distance < threshold, else 'far'.
    
    Parameters:
    - df: DataFrame with 'Latitude', 'Longitude', and 'Assigned_RepZone' columns
    - repzone_offices_adresses: Dictionary mapping office names to (lat, lon) coordinates
    - threshold_km: Distance threshold to classify as 'near' or 'far'
    
    Returns:
    - Modified DataFrame with two new columns.
    """
    
    def compute_distance(row):
        centroid = (row["Latitude"], row["Longitude"])
        office_coords = repzone_offices_adresses[row["Assigned_RepZone"]]
        return geodesic(centroid, office_coords).km

    df = df.copy()
    df["Distance_to_RepZone_km"] = df.apply(compute_distance, axis=1)
    df["RepZone_Proximity"] = np.where(df["Distance_to_RepZone_km"] < threshold_km, "near", "far")
    print('Debugging inside the function')
    return df


def solve_weekly_routes_far(real_durations,customer_ids, num_vehicles, SERVICE_TIME=30, max_work_time=480):
    '''
    Solves the routing problem for one cluster-week using OR-Tools.

    Input:
    ------
    
    real_durations (np.array): Duration matrix including RepZone at index 0.
    
    customer_ids (list): Original customer IDs, where customer_ids[0] == 'RepZone'.
    
    num_vehicles (int): Number of reps * days for this schedule.
    
    SERVICE_TIME (int): Service time at each stop in minutes.
    
    max_work_time (int): Max total work time per vehicle (in minutes).

    Output:
    -------
    
    dict: vehicle_id → list of CustomerIds (route)
    
    '''


    data = {
        'time_matrix': real_durations.tolist(),
        'num_vehicles': num_vehicles,
        'depot': 0
    }

    manager = pywrapcp.RoutingIndexManager(
        len(data['time_matrix']),
        data['num_vehicles'],
        [data['depot']] * data['num_vehicles'],
        [data['depot']] * data['num_vehicles']
    )

    routing = pywrapcp.RoutingModel(manager)

    def total_time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data['time_matrix'][from_node][to_node]
        service_time = 0 if from_node == 0 else SERVICE_TIME
        return travel_time + service_time

    total_time_callback_index = routing.RegisterTransitCallback(total_time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(total_time_callback_index)

 
    # Add time dimension
    routing.AddDimension(
        total_time_callback_index,
        1200,              # slack
        15000,     # hard upper bound, very large
        True,              # start cumul at zero
        "Time"
    )

    time_dimension = routing.GetDimensionOrDie("Time")
    # time_dimension.SetGlobalSpanCostCoefficient(1000)

    


 
    

    for vehicle_id in range(num_vehicles):
        end_index = routing.End(vehicle_id)
        start_index = routing.Start(vehicle_id)

        time_dimension.SetCumulVarSoftUpperBound(end_index, max_work_time, 1000)
        

        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(end_index))
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(start_index))

     

    # Hard constraint: each customer visited once
    penalty = 75000
    for customer_idx in range(1, len(data['time_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(customer_idx)], penalty)

    # Loosened VisitCount dimension
    visit_callback_index = routing.RegisterUnaryTransitCallback(lambda index: 1)
    routing.AddDimension(
        visit_callback_index,
        0,                  # no slack
        30,                 # allow many visits per rep, just soft-balance
        True,
        "VisitCount"
    )
    visit_dimension = routing.GetDimensionOrDie("VisitCount")
    for vehicle_id in range(num_vehicles):
        end_index = routing.End(vehicle_id)
        # Encourage soft upper limit of 3 visits per rep
        visit_dimension.SetCumulVarSoftUpperBound(end_index, 3, 500)
        visit_dimension.CumulVar(end_index).SetMax(6)



    # Search
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 150

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        print("No feasible solution found.")
        return None

    routes = {}
    new_vehicle_id = 0

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append('RepZone' if node == 0 else customer_ids[node])
            index = solution.Value(routing.NextVar(index))

        if len(route) > 1:
            routes[new_vehicle_id] = route
            new_vehicle_id += 1

    return routes
def run_clustering_pipeline(json_customer_data):
    df_Customer_Table = pd.DataFrame(json_customer_data)

    # Project to UTM (EPSG:32633)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    utm_coords = np.array([
        transformer.transform(lon, lat) for lat, lon in zip(df_Customer_Table['Latitude'], df_Customer_Table['Longitude'])
    ])

    inertias = calculate_inertias(utm_coords)
    best_k = auto_select_k(inertias, (5, 30))
    df_balanced = balanced_kmeans_clustering(df_Customer_Table, initial_k=best_k)

    df_centroids = df_balanced.groupby('cluster').agg({'Latitude': 'mean', 'Longitude': 'mean'}).reset_index()
    df_centroids['Assigned_RepZone'] = df_centroids.apply(
        lambda row: get_closest_repzone(row['Latitude'], row['Longitude']), axis=1
    )
    df_balanced = df_balanced.merge(df_centroids[['cluster', 'Assigned_RepZone']], on='cluster')

    repzone_coords = [(lon, lat) for lat, lon in repzone_offices.values()]
    df_balanced = df_balanced[df_balanced.apply(lambda row: not is_too_far(row, repzone_coords), axis=1)]

    # Reproject to EPSG:5254
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5254", always_xy=True)
    utm_coords = np.array([
        transformer.transform(lon, lat) for lat, lon in zip(df_balanced['Latitude'], df_balanced['Longitude'])
    ])
    inertias = calculate_inertias(utm_coords)
    best_k = auto_select_k(inertias, (5, 30))
    df_balanced = balanced_kmeans_clustering(df_balanced, initial_k=best_k)

    df_centroids = df_balanced.groupby('cluster').agg({'Latitude': 'mean', 'Longitude': 'mean'}).reset_index()
    df_centroids['Assigned_RepZone'] = df_centroids.apply(
        lambda row: get_closest_repzone(row['Latitude'], row['Longitude']), axis=1
    )
    df_balanced = df_balanced.merge(df_centroids[['cluster', 'Assigned_RepZone']], on='cluster')

    return df_balanced


def plot_cluster_routes(cluster_key, routes_all, df_schedule, df_cluster_summary, repzone_offices):
    # Extract metadata
    parts = cluster_key.split('_')
    cluster_id = int(parts[1])
    week = int(parts[3])
    routes = routes_all[cluster_key]

    # Get RepZone name and lat/lon
    repzone_name = df_cluster_summary[df_cluster_summary["cluster"] == cluster_id]["Assigned_RepZone"].values[0]
    repzone_latlon = tuple(repzone_offices[repzone_name])

    # Filter schedule for that week/cluster
    df_week = df_schedule[(df_schedule["week"] == week) & (df_schedule["cluster"] == cluster_id)]
    coord_lookup = df_week.set_index("CustomerId")[["Latitude", "Longitude"]].to_dict("index")
    coord_lookup["RepZone"] = {"Latitude": repzone_latlon[0], "Longitude": repzone_latlon[1]}

    # Initialize map
    m = folium.Map(location=repzone_latlon, zoom_start=10)
    colors = get_cmap('tab20', len(routes))

    # Loop through vehicles
    for vehicle_id, route_ids in routes.items():
        route_group = folium.FeatureGroup(name=f"Vehicle {vehicle_id}")
        route_coords = []
        # RepZone coordinates
        repzone_coord = (coord_lookup["RepZone"]["Latitude"], coord_lookup["RepZone"]["Longitude"])
        
        # Start route with RepZone (but do NOT add marker here — handled globally)
        route_coords = [repzone_coord]
        visit_order = 0
        
        # Loop over remaining client IDs
        for cid in route_ids[1:]:  # skip 'RepZone' literal
            lat = coord_lookup[cid]["Latitude"]
            lon = coord_lookup[cid]["Longitude"]
            coord = (lat, lon)
            route_coords.append(coord)
        
            # Plot client stop marker
            CircleMarker(
                location=coord,
                radius=4,
                color='black',
                fill=True,
                fill_opacity=0.8,
                tooltip=f"Vehicle {vehicle_id} - Stop #{visit_order} - {cid}"
            ).add_to(route_group)
            visit_order += 1

        if len(route_coords) > 1:
            road_path = fetch_ors_directions(route_coords)
            if road_path:
                color = "#{:02x}{:02x}{:02x}".format(*[int(255 * c) for c in colors(int(vehicle_id) % 20)[:3]])
                PolyLine(road_path, color=color, weight=4, tooltip=f"Vehicle {vehicle_id}").add_to(route_group)

                Marker(
                    location=road_path[0],
                    popup="RepZone Start",
                    tooltip="RepZone Start",
                    icon=Icon(color='green', icon='building', prefix='fa')  # or 'home', 'flag', etc.
                ).add_to(route_group)
                Marker(road_path[-1], popup="End", icon=Icon(color='red')).add_to(route_group)

        route_group.add_to(m)

    LayerControl().add_to(m)
    return m
