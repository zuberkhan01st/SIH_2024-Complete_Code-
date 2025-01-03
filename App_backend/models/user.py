from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token

mongo = PyMongo()
bcrypt = Bcrypt()

# Function to create a hashed password
def hash_password(password):
    return bcrypt.generate_password_hash(password).decode('utf-8')

# Function to check if password matches
def check_password(hashed_password, password):
    return bcrypt.check_password_hash(hashed_password, password)

# Function to create JWT token
def create_token(user_id):
    return create_access_token(identity=str(user_id))

# Function to get a user by username
def get_user_by_username(username):
    user = mongo.db.users.find_one({"username": username})
    return user
