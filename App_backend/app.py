from flask import Flask
from flask_jwt_extended import JWTManager
from db import mongo  # Import mongo from db.py
from routes.auth import auth
from routes.alerts import alerts
import os
from dotenv import load_dotenv
from flask_cors import CORS  # This will allow all domains to access your API


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

CORS(app)

@app.route('/')
def main():
    return "Working!"

# Load the environment variables from the .env file
app.config["MONGO_URI"] = os.getenv("MONGO_URI")  # Get the Mongo URI from .env
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")  # Get JWT secret key from .env

# Initialize the extensions
mongo.init_app(app)
jwt = JWTManager(app)

# Register blueprints
app.register_blueprint(auth, url_prefix='/api/auth')
app.register_blueprint(alerts, url_prefix='/api/alerts')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
