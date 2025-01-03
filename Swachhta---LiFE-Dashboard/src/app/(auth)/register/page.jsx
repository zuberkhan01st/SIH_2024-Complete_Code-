"use client";

import React, { useState } from "react";
import Link from "next/link";
import axios from "axios";
import { useRouter } from "next/navigation";
import { signIn } from "next-auth/react";
import Image from "next/image";

export default function SignUp() {
  const router = useRouter();

  const [loading, setLoading] = useState(false);
  const [userState, setUserState] = useState({
    email: "",
    password: "",
    name: "",
    password_confirmation: "",
  });

  const [errors, setError] = useState({});

  const submitForm = async () => {
    setLoading(true);
    try {
      const res = await axios.post("/api/auth/register", userState);
      const response = res.data;
      if (response.status === 200) {
        router.push(`/login?message=${response.msg}`);
      } else if (response.status === 400) {
        setError(response.errors);
      } else {
        setError({});
      }
    } catch (err) {
      console.log("Error is", err);
    } finally {
      setLoading(false);
    }
  };

  const googleLogin = async () => {
    await signIn("google", {
      callbackUrl: "/dashboard",
      redirect: true,
    });
  };

  return (
    <section className="bg-gray-50 dark:bg-gray-900 min-h-screen flex items-center justify-center">
      <div className="w-full max-w-md p-8 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Sign Up
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          Already have an account?
          <Link
            href="/login"
            className="font-medium text-blue-500 hover:underline ml-2"
          >
            Sign In
          </Link>
        </p>
        <form onSubmit={(e) => e.preventDefault()} className="space-y-5">
          <div>
            <label htmlFor="name" className="block text-gray-700 dark:text-gray-300">
              Full Name
            </label>
            <input
              id="name"
              type="text"
              placeholder="Full Name"
              className="mt-2 w-full h-12 px-4 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200"
              onChange={(e) => setUserState({ ...userState, name: e.target.value })}
            />
            {errors.name && <p className="text-red-500 mt-1">{errors.name}</p>}
          </div>
          <div>
            <label htmlFor="email" className="block text-gray-700 dark:text-gray-300">
              Email address
            </label>
            <input
              id="email"
              type="email"
              placeholder="Email"
              className="mt-2 w-full h-12 px-4 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200"
              onChange={(e) => setUserState({ ...userState, email: e.target.value })}
            />
            {errors.email && <p className="text-red-500 mt-1">{errors.email}</p>}
          </div>
          <div>
            <label htmlFor="password" className="block text-gray-700 dark:text-gray-300">
              Password
            </label>
            <input
              id="password"
              type="password"
              placeholder="Password"
              className="mt-2 w-full h-12 px-4 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200"
              onChange={(e) => setUserState({ ...userState, password: e.target.value })}
            />
            {errors.password && <p className="text-red-500 mt-1">{errors.password}</p>}
          </div>
          <div>
            <label htmlFor="password_confirmation" className="block text-gray-700 dark:text-gray-300">
              Confirm Password
            </label>
            <input
              id="password_confirmation"
              type="password"
              placeholder="Confirm Password"
              className="mt-2 w-full h-12 px-4 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200"
              onChange={(e) =>
                setUserState({ ...userState, password_confirmation: e.target.value })
              }
            />
          </div>
          <button
            type="button"
            className={`w-full py-3 rounded-md text-white font-semibold ${loading ? 'bg-gray-600' : 'bg-blue-600'} hover:bg-blue-700 transition duration-300`}
            onClick={submitForm}
            disabled={loading}
          >
            {loading ? "Processing..." : "Create Account"}
          </button>
        </form>
        <div className="my-3 text-center text-gray-600 dark:text-gray-400">OR</div>
        <div className="space-y-4">
          <button
            type="button"
            className="w-full py-3 rounded-md border border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 text-gray-900 dark:text-gray-100 font-semibold flex items-center justify-center hover:bg-gray-100 dark:hover:bg-gray-600 transition duration-300"
            onClick={googleLogin}
          >
            <Image
              src="/google_icon.png"
              height={30}
              width={30}
              alt="Google Icon"
              className="mr-3"
            />
            Sign in with Google
          </button>
        </div>
      </div>
    </section>
  );
}
