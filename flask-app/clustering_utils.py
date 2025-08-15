# clustering_logic.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pyproj import Transformer
from kneed import KneeLocator
from haversine import haversine, Unit
from dotenv import load_dotenv
import os
from geopy.distance import geodesic
import requests
import logging
import traceback
load_dotenv()  # Loads from .env by default
#Get the end point of ors-app ,it is available in docker-compose.yml file in flask service
ORS_MATRIX_URL = os.getenv("ORS_API_URL")
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

def balanced_kmeans_clustering(
    customer_data,
    lat_key='Latitude',
    lon_key='Longitude',
    min_size=30,
    max_size=300,
    initial_k=10,
    random_state=42,
):
    '''
    Function to calculate clusters from customer data JSON.
    
    Input:
    -----
    customer_data: List of dictionaries containing customer data
    lat_key: Key for latitude, default is 'Latitude'
    lon_key: Key for longitude, default is 'Longitude'
    min_size: Default size is 30
    max_size: Default size is 300
    initial_k: Default is 10
    
    Output:
    ------
    List of dictionaries with original data plus cluster assignments
    '''
    # Create DataFrame from customer_data list
    df = pd.DataFrame(customer_data)
    
    # Check if required columns exist
    if lat_key not in df.columns or lon_key not in df.columns:
        raise ValueError(f"Missing required columns: {lat_key} and/or {lon_key}")
    
    # Convert lat/lon to float if they're not already
    df[lat_key] = pd.to_numeric(df[lat_key], errors='coerce')
    df[lon_key] = pd.to_numeric(df[lon_key], errors='coerce')
    
    # Drop rows with NaN coordinates
    df = df.dropna(subset=[lat_key, lon_key])
    
    # Step 1: Project coordinates to UTM coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)
    utm_coords = np.array([
        transformer.transform(lon, lat) for lat, lon in zip(df[lon_key], df[lat_key])
    ])
    df['x'] = utm_coords[:, 0]
    df['y'] = utm_coords[:, 1]

    # Step 2: Initial KMeans clustering
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

            elif size < min_size and len(cluster_sizes) > 1:  # Ensure we have more than one cluster
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

    # Convert cluster to string for consistent handling
    df['cluster'] = df['cluster'].astype(str)
    
    # Calculate cluster centers for later use
    cluster_centers = {}
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]
        cluster_centers[cluster_id] = {
            'lat': cluster_data[lat_key].mean(),
            'lon': cluster_data[lon_key].mean(),
            'count': len(cluster_data)
        }
    
    # Drop temporary columns and convert back to list of dictionaries
    df = df.drop(columns=['x', 'y'])
    
    # Return both clustered data and centers
    return df.to_dict('records'), cluster_centers
def get_ors_duration_distance_clust(origins, destinations):
    """
    Calculate the distance and travel duration using ORS matrix API.
    """
    # Combine origins and destinations coordinates
    locations = origins + destinations

    # Create body for the matrix request
    body = {
        "locations": locations,
        "metrics": ["duration", "distance"], # Metrics to include in the response
    }

    logging.info("Sending request to ORS matrix endpoint: %s", body)

    try:
        # Make the POST request to ORS Matrix API
        response = requests.post(ORS_MATRIX_URL, json=body)
        logging.info("ORS response status: %d", response.status_code)
        logging.debug("ORS response text: %s", response.text)

        # If the response is successful
        if response.status_code == 200:
            data = response.json()

            # Extract durations and distances
            durations = data["durations"]  # Duration in minutes
            distances = data["distances"]  # Distance in kilometers

            # Return the durations and distances
            return durations, distances, None
        else:
            # Return the error message from ORS if status code is not 200
            logging.error("Error from ORS API: %s", response.text)
            return None, None, response.text
    
    except requests.exceptions.RequestException as e:
        logging.error("RequestException occurred while contacting ORS: %s", str(e))
        traceback.print_exc()
        return None, None, str(e)
    except Exception as e:
        logging.error("Exception occurred while contacting ORS: %s", str(e))
        traceback.print_exc()
        return None, None, str(e)
def process_customer_data_for_clustering_clust(customer_data, min_size=30, max_size=300):
    """
    Process customer data and apply clustering
    
    Input:
    -----
    customer_data: List of dictionaries containing customer location data
    min_size: Minimum cluster size
    max_size: Maximum cluster size
    
    Output:
    ------
    Tuple of (clustered_data, cluster_centers)
    """
    if not customer_data:
        return [], {}
    
    # Find optimal k if we have enough data points
    df = pd.DataFrame(customer_data)
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Project coordinates for finding optimal k
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)
    utm_coords = np.array([
        transformer.transform(lon, lat) for lat, lon in zip(df['Longitude'], df['Latitude'])
    ])
    
    # Determine optimal initial k if we have enough data points
    if len(df) >= 10:
        k_range = (2, min(30, len(df) // 2))  # Reasonable range based on data size
        inertias = calculate_inertias(utm_coords, k_range)
        best_k = auto_select_k(inertias, k_range) or 5  # Default to 5 if auto-selection fails
    else:
        best_k = min(5, len(df))  # For small datasets
    
    # Apply clustering with balanced sizes
    min_cluster_size = min(min_size, max(5, len(df) // 10))  # Adjust min size based on data
    max_cluster_size = min(max_size, len(df))  # Ensure max size isn't larger than dataset
    
    return balanced_kmeans_clustering(
        customer_data,
        min_size=min_cluster_size,
        max_size=max_cluster_size,
        initial_k=best_k
    )

def get_closest_repzone_clust(client_lat, client_lon):
    distances = {}
  
    for office, coords in repzone_offices.items():
        dist = geodesic((client_lat, client_lon), tuple(coords)).km  # distance in km
        distances[office] = dist
    closest_office = min(distances, key=distances.get)
    return closest_office

