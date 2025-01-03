import mongoose, { Schema } from "mongoose";

const userSchema = new Schema({
  name: {
    required: [true, "Name field is required."],
    minLength: [2, "Name must be 2 characters long."],
    type: String,
  },
  email: {
    required: [true, "Email field is required."],
    type: String,
    unique: true,
    trim: true,
  },
  password: {
    type: String,
  },
  avtar: {
    required: false,
    type: String,
  },
  role: {
    required: true,
    type: String,
    default: "User",
  },
  password_reset_token: {
    required: false,
    type: String,
    trim: true,
  },
  magic_link_token: {
    required: false,
    type: String,
    trim: true,
  },
  magic_link_sent_at: {
    required: false,
    type: Date,
  },
});

export const User = mongoose.models.User || mongoose.model("User", userSchema);
