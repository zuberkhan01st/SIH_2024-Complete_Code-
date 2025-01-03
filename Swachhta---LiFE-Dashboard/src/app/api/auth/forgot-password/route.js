import { User } from "@/models/User";
import { NextRequest, NextResponse } from "next/server";
import cryptoRandomString from "crypto-random-string";
import Cryptr from "cryptr";
import Env from "@/config/env";
import { render } from "@react-email/render";
import ForgotPasswordEmail from "@/emails/ForgotPasswordEmail";
import { sendEmail } from "@/config/mail";
import { connect } from "@/database/mongo.config";

// Ensure the database connection is established
connect();

export async function POST(request) {
  const payload = await request.json();

  // Check if the user with the provided email exists
  const user = await User.findOne({ email: payload.email });
  if (!user) {
    return NextResponse.json({
      status: 400,
      errors: {
        email: "No user found with this email.",
      },
    });
  }

  // Generate a random string for the password reset token
  const randomStr = cryptoRandomString({
    length: 64,
    type: "alphanumeric",
  });

  user.password_reset_token = randomStr;
  await user.save();

  // Encrypt the user email
  const crypt = new Cryptr(Env.SECRET_KEY);
  const encryptedEmail = crypt.encrypt(user.email);

  const url = `${Env.APP_URL}/reset-password/${encryptedEmail}?signature=${randomStr}`;

  try {
    const html = render(
      ForgotPasswordEmail({
        params: {
          name: user.name,
          url: url,
        },
      })
    );

    // Send the password reset email
    await sendEmail(payload.email, "Reset Password", html);
    return NextResponse.json({
      status: 200,
      message: "Email sent successfully. Please check your email.",
    });
  } catch (error) {
    console.log("The error is", error);
    return NextResponse.json({
      status: 500,
      message: "Something went wrong. Please try again!",
    });
  }
}
