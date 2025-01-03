import { connect } from '@/database/mongo.config';
import { User } from '@/models/User';

connect(); // Ensure database connection

export async function GET() {

  try {
    const users = await User.find({}, '_id name email'); // Fetch _id, name, and email fields
    const userDetails = users.map(user => ({
      id: user._id.toString(),
      name: user.name,
      email: user.email,
    }));
    return new Response(JSON.stringify({ userDetails }), { status: 200 });
  } catch (error) {
    console.error('Error fetching user details:', error);
    return new Response(JSON.stringify({ error: 'Failed to fetch user details' }), { status: 500 });
  }
}
