import { connect } from '@/database/mongo.config';
import { User } from '@/models/User';
import { sendEmail } from '@/config/mail';

export async function POST(req) {
  await connect(); // Ensure database connection

  try {
    const { userId, message } = await req.json(); // Parse incoming JSON data
    const user = await User.findById(userId).select('email'); // Find user by ID and get email

    if (!user) {
      return new Response(JSON.stringify({ error: 'User not found' }), { status: 404 });
    }

    const emailSent = await sendEmail(
      user.email,
      'Message from Admin',
      `<p>${message}</p>` // HTML content for email
    );

    if (emailSent) {
      return new Response(JSON.stringify({ success: 'Message sent successfully' }), { status: 200 });
    } else {
      return new Response(JSON.stringify({ error: 'Failed to send email' }), { status: 500 });
    }
  } catch (error) {
    console.error('Error sending message:', error);
    return new Response(JSON.stringify({ error: 'Failed to send message' }), { status: 500 });
  }
}
