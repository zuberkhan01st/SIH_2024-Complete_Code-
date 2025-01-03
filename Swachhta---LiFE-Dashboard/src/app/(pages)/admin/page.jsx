"use client";
import React, { useState } from "react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
import Toast from "@/components/Toast";
import Link from "next/link";

export default function AdminLogin() {
  const router = useRouter();
  const [authState, setAuthState] = useState({
    email: "",
    password: "",
  });

  const handleSubmit = async (event) => {
    event.preventDefault();
    const data = await signIn("credentials", {
      email: authState.email,
      password: authState.password,
      redirect: false,
    });

    if (data?.status === 200) {
      router.replace("/admin/dashboard");
    } else {
      // Handle errors or show messages
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-100 dark:bg-gray-900">
      <Toast />
      <div className="w-full max-w-md bg-white dark:bg-gray-800 shadow-lg rounded-lg p-6">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Admin Login</h1>
        <p className="text-gray-600 dark:text-gray-400">Welcome back</p>
        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          <div>
            <label className="block text-gray-700 dark:text-gray-300">Email</label>
            <input
              type="text"
              placeholder="Enter your email"
              className="w-full border-gray-300 dark:border-gray-700 p-2 h-10 rounded-md border focus:ring-red-500 dark:focus:ring-red-500"
              onChange={(e) =>
                setAuthState({ ...authState, email: e.target.value })
              }
            />
          </div>
          <div>
            <label className="block text-gray-700 dark:text-gray-300">Password</label>
            <input
              type="password"
              placeholder="Enter your password"
              className="w-full border-gray-300 dark:border-gray-700 p-2 h-10 rounded-md border focus:ring-red-500 dark:focus:ring-red-500"
              onChange={(e) =>
                setAuthState({ ...authState, password: e.target.value })
              }
            />
          </div>
          <div className="flex flex-col gap-3">
            <button
              type="submit"
              className="w-full bg-green-500 hover:bg-green-600 dark:bg-green-400 dark:hover:bg-green-500 rounded-lg p-2 text-white"
            >
              Login
            </button>
            <Link href='/'>
            <button
              type="submit"
              className="w-full bg-red-500 hover:bg-red-600 dark:bg-red-400 dark:hover:bg-red-500 rounded-lg p-2 text-white"
            >
              Back
            </button>
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}
