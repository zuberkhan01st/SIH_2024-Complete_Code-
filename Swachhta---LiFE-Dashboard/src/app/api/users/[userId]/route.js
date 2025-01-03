import { connect } from '@/database/mongo.config';
import { User } from '@/models/User';

export async function GET(req, { params }) {
  await connect(); // Ensure database connection

  try {
    const { userId } = params; // Extract userId from params
    const user = await User.findById(userId).select('name email'); // Fetch user data

    if (!user) {
      return new Response(JSON.stringify({ error: 'User not found' }), { status: 404 });
    }

    return new Response(JSON.stringify({ success: true, user }), { status: 200 });
  } catch (error) {
    console.error('Error fetching user:', error);
    return new Response(JSON.stringify({ error: 'Failed to fetch user' }), { status: 500 });
  }
}
