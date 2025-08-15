# NOTE :
  The current Repository has only flask-app dependencies and it does not contains ors-app dependencies for complete installation follow the **setup instruction**
# 🗺️ OpenRouteTurkey - Distance Matrix Service and OR-Tools Route Optimization

This project integrates a Docker image of **OpenRouteService (ORS)** with a **Flask** API to provide a simple **distance and duration matrix calculator**. In addition, it uses Google's **OR-Tools** to solve the **Vehicle Routing Problem (VRP)**.

You can send a POST request with coordinates, and the Flask service will return distance and duration data between origins and destinations using the internal ORS Docker container.

---

## ⚙️ Setup Instructions


### 📥 Download

You can download the latest version of the app from the link below:

🔗 [Download App](https://drive.google.com/file/d/1Cc8Z5ajY1jYCo9BOdEuv5vnos6kA1F6r/view?usp=sharing)

---

### 🛠 Installation Guide

For step-by-step installation instructions, refer to the guide below:

📄 [View Installation Guide](https://drive.google.com/file/d/1OWdsTkBFkjCEPObNXJZCGIQlMydChTcB/view?usp=drive_link)

### 3. Wait for ORS to be Ready
```yml
http://localhost:8084/ors/v2/health

```
### 4. Test the api
```yml
curl -X POST http://localhost:5000/get_optimized_root \
  -H "Content-Type: application/json" \
  -d '{
    "origins": [[28.9784, 41.0082]],
    "destinations": [[34.0522, 38.4192]]
  }'
```
## For development Guide
### Project Structure
```yml
OPENROUTETURKEY/
├── flask-app/
│   ├── __pycache__/
│   ├── static/
│   │   └── maps/
│   │       ├── customer_table.json
│   │       ├── desktop.ini
│   │       ├── df_centroids.json
│   │       ├── df_schedule.json
│   │       ├── k.json
│   │       ├── lipynb
│   │       ├── office_locations.json
│   │       └── results.json
│   ├── templates/
│   │   ├── .txt
│   │   ├── clustering_ui.html
│   │   └── index.html
│   ├── app.py
│   ├── clustering_utils.py
│   ├── sec_utils.py
│   └── Dockerfile
├── ors-docker/
│   ├── requirements.txt
│   └── docker-compose.yml
├── Readme.md   
```
### 🌐 HTML / Frontend

All HTML templates go in templates/
All static files (CSS, JS, etc.) go in flask-app/static/
### 🛠 Add New Environment Variable / API
Go to ors-docker/docker-compose.yml
Under flask → environment, add:
```yml
- YOUR_KEY=your_value
```
Access in flask
```yml
value = os.environ.get('YOUR_KEY')
```
## 📦 OR-Tools Optimization Engine

This project integrates [Google OR-Tools](https://developers.google.com/optimization) to solve the **Vehicle Routing Problem (VRP)** using real world distances obtained from OpenRouteService for clients across Turkey.

We first divide the clients into geographic clusters using Kmeans, then take frequency into consideration as specific weekly or monthly visits.

We use OR-Tools as a constraint solver to optimize routes under several parameters:

- Minimize total distance traveled across all vehicles
- Soft balance workload across representatives
- Start routes at a fixed depot (RepZone office)

The optimization runs inside the Flask API and returns routes that improve over manual assignments.

As an example on how to interact with the OR-Tools result with the flask application:

📄 [View Results in Flask API](https://drive.google.com/file/d/1IBJCZGNoMan0fL_Gzojlec_O0MVqx2dB/view?usp=drive_link)

## 📓 Jupyter Notebook

A complete modeling workflow is provided in the Jupyter notebook located in the [`notebooks`](./notebooks) folder of this repository.  
It includes data preprocessing, route optimization using OR-Tools, and result visualization steps.
