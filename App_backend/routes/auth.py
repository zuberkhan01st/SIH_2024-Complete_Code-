# routes/auth.py
from flask import Blueprint, request, jsonify
from models.user import hash_password, check_password, create_token
from db import mongo  # Import mongo from db.py


auth = Blueprint('auth', __name__)


@auth.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    user_type = data.get('user_type')
    email = data.get('email')

    if not username or not password:
        return jsonify({"error": "Username and password required!"}), 400

    user = mongo.db.users.find_one({"username": username})

    if user:
        return jsonify({"error": "User already exists!"}), 400

    hashed_password = hash_password(password)

    new_user = {
        "username": username,
        "password": hashed_password,
        "user_type":user_type,
        "email":email
    }
    
    mongo.db.users.insert_one(new_user)
    return jsonify({"message": "User created successfully!"}), 201

@auth.route('/login', methods=['POST'])
def login():
    print("Received login request")
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password required!"}), 400

    user = mongo.db.users.find_one({"username": username})

    if not user or not check_password(user['password'], password):
        return jsonify({"error": "Invalid credentials!"}), 401

    token = create_token(user['_id'])
    print("Login successful")
    return jsonify({"message": "Login successful!", "token": token}), 200
