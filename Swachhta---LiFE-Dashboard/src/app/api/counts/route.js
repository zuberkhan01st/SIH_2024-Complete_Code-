import { Alert } from "@/models/alert"; // Import the Alert model
import { User } from "@/models/User";   // Import the User model
import { connect } from "@/database/mongo.config"; // Database connection

export const GET = async (req) => {
  const headers = new Headers();
  setCorsHeaders(headers);

  try {
    await connect(); // Connect to the database

    // Fetch all users
    const users = await User.find();

    // Fetch alert count for each user
    const userAlertsCount = await Promise.all(
      users.map(async (user) => {
        // Count alerts for each user where status is not resolved
        const alertCount = await Alert.countDocuments({ assignedTo: user._id, status: { $ne: 'Resolved' } });
        
        return {
          userId: user._id,
          name: user.name,
          alertCount,  // Alert count for the user
        };
      })
    );

    // Return the user data with alert counts
    return new Response(
      JSON.stringify({ success: true, userAlertsCount }),
      { headers, status: 200 }
    );
  } catch (error) {
    console.error('Error fetching user alerts count:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to fetch user alerts count.' }),
      { headers, status: 500 }
    );
  }
};

// Helper function to set CORS headers
const setCorsHeaders = (headers) => {
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
};
