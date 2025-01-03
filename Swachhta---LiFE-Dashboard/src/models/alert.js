import mongoose, { Schema } from "mongoose";

const alertSchema = new Schema({
  message: {
    type: String,
    required: [true, "Message is required."],
  },
  area: {
    type: String,
    required: [true, "Area is required."],
  },
  issuedDate: {
    type: Date,
    default: Date.now,
  },
  status: {
    type: String,
    enum: ["Pending", "In Progress", "Resolved"],
    default: "Pending",
  },
  details: {
    type: String,
  },
  assignedTo: {
    type: Schema.Types.ObjectId, 
    ref: "User",
    required: true,
  },
  updates: [
    {
      date: { type: Date, default: Date.now },
      comment: { type: String },
    },
  ],
});

export const Alert = mongoose.models.Alert || mongoose.model("Alert", alertSchema);
