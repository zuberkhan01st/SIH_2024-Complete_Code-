from flask import Flask, Blueprint, request, jsonify, Response
from db import mongo
from bson import ObjectId
import json
import time

# Initialize Flask app
app = Flask(__name__)

# Create a Blueprint for the alerts API
alerts = Blueprint('alerts', __name__)

# Route to create a new alert
@alerts.route('/create', methods=['POST'])
def create_alert():
    data = request.get_json()

    # Extract alert data from the request body
    title = data.get('title')
    description = data.get('description')
    created_by = data.get('created_by')
    location = data.get('location')
    post_id = data.get('post_id')

    # Check if all required fields are provided
    if not title or not description or not created_by or not location:
        return jsonify({"error": "Missing required fields"}), 400

    # Create an alert object
    alert = {
        "title": title,
        "description": description,
        "created_by": created_by,
        "location": location,
        "status": "Active",  # Status of the alert (could be "active" or "resolved")
        "post_id": post_id
    }

    # Insert the alert into the MongoDB collection
    mongo.db.alerts.insert_one(alert)

    return jsonify({"message": "Alert created successfully!"}), 201

# Route to fetch all active alerts
@alerts.route('/', methods=['GET'])
def get_alerts():
    # Fetch all active alerts from the MongoDB collection
    active_alerts = mongo.db.alerts.find({"status": "Active"})

    alerts_list = []
    for alert in active_alerts:
        alert['_id'] = str(alert['_id'])  # Convert ObjectId to string for JSON serialization
        alerts_list.append(alert)

    return jsonify(alerts_list), 200

# Server-Sent Events for real-time updates
@alerts.route('/stream', methods=['GET'])
def stream_alerts():
    def generate():
        last_alert_id = None
        while True:
            # Fetch the next active alert after the last alert's ID
            alert = mongo.db.alerts.find_one(
                {"_id": {"$gt": last_alert_id}, "status": "Active"},
                sort=[('_id', 1)]
            )
            if alert:
                last_alert_id = alert['_id']
                alert['_id'] = str(alert['_id'])  # Convert ObjectId to string for JSON serialization
                yield f"data: {json.dumps(alert)}\n\n"
            time.sleep(1)  # Check every second for new alerts

    return Response(generate(), content_type='text/event-stream')

# Route to fetch active alerts by a specific post ID
@alerts.route('/post/<post_id>', methods=['GET'])
def get_alerts_by_post(post_id):
    # Fetch active alerts related to a specific post ID
    active_alerts = mongo.db.alerts.find({"status": "Active", "post_id": post_id})

    alerts_list = []
    for alert in active_alerts:
        alert['_id'] = str(alert['_id'])  # Convert ObjectId to string for JSON serialization
        alerts_list.append(alert)

    if not alerts_list:
        return jsonify({"message": "No active alerts found for this post ID."}), 404

    return jsonify(alerts_list), 200

# Route to resolve an alert (change its status to "resolved")
@alerts.route('/resolve/<alert_id>', methods=['PUT'])
def resolve_alert(alert_id):
    try:
        # Convert the alert_id to an ObjectId
        alert_object_id = ObjectId(alert_id)
    except Exception as e:
        return jsonify({"error": "Invalid alert ID"}), 400

    # Find the alert by its ID and update its status to "Resolved"
    result = mongo.db.alerts.update_one(
        {"_id": alert_object_id},
        {"$set": {"status": "Resolved"}}
    )

    if result.modified_count == 0:
        return jsonify({"error": "Alert not found or already resolved"}), 404

    return jsonify({"message": "Alert resolved successfully!"}), 200

# Register the alerts Blueprint
app.register_blueprint(alerts, url_prefix='/api/alerts')

# Main route
@app.route('/')
def home():
    return "Welcome to the Alerts API!"

if __name__ == '__main__':
    app.run(debug=True)
