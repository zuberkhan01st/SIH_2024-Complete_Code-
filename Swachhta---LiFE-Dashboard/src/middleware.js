import { getToken } from "next-auth/jwt";
import { NextRequest, NextResponse } from "next/server";

export async function middleware(request) {
  const { pathname } = request.nextUrl;

  // Allow public access to login routes
  if (pathname === "/login" || pathname === "/admin/login") {
    return NextResponse.next();
  }

  // Fetch the token
  const token = await getToken({ req: request });

  // Define protected route patterns for users and admins
  const userProtectedRoutes = ["/dashboard"];
  const adminProtectedRoutes = ["/admin/dashboard"];

  // Redirect to appropriate login page if the user is not authenticated and tries to access protected routes
  if (token == null) {
    if (userProtectedRoutes.some(route => pathname.startsWith(route))) {
      return NextResponse.redirect(
        new URL(
          "/login?error=Please login first to access this route",
          request.url
        )
      );
    }

    if (adminProtectedRoutes.some(route => pathname.startsWith(route))) {
      return NextResponse.redirect(
        new URL(
          "/admin",
          request.url
        )
      );
    }
  }

  // Extract the user from the token
  const user = token?.user;

  // Redirect to admin login if a user tries to access admin routes
  if (
    adminProtectedRoutes.some(route => pathname.startsWith(route)) &&
    user?.role === "User"
  ) {
    return NextResponse.redirect(
      new URL(
        "/admin/login?error=Please login first to access this route.",
        request.url
      )
    );
  }

  // Redirect to user login if an admin tries to access user routes
  if (
    userProtectedRoutes.some(route => pathname.startsWith(route)) &&
    user?.role === "Admin"
  ) {
    return NextResponse.redirect(
      new URL(
        "/login?error=Please login first to access this route.",
        request.url
      )
    );
  }

  // Allow access if no conditions matched
  return NextResponse.next();
}
