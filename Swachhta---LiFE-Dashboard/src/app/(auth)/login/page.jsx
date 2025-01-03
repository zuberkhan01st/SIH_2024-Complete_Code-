"use client";
import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import axios from "axios";
import { signIn } from "next-auth/react";
import Image from "next/image";
import Toast from "@/components/Toast";

export default function SignInOne() {
  const searchParam = useSearchParams();

  const [authData, setAuthData] = useState({
    email: "",
    password: "",
  });
  const [loading, setLoading] = useState(false);
  const [errors, setError] = useState({});

  useEffect(() => {
    console.log("The query is", searchParam.get("error"));
  }, [searchParam]);

  const submitForm = async () => {
    setLoading(true);
    try {
      const res = await axios.post("/api/auth/login", authData);
      const response = res.data;
      if (response.status === 200) {
        signIn("credentials", {
          email: authData.email,
          password: authData.password,
          callbackUrl: "/dashboard",
          redirect: true,
        });
      } else if (response.status === 400) {
        setError(response.errors);
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
      <Toast />
      <div className="w-full max-w-md p-8 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Sign In
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          Please enter your email and password to continue.
        </p>
        <form onSubmit={(e) => e.preventDefault()} className="space-y-5">
          <div>
            <label htmlFor="email" className="block text-gray-700 dark:text-gray-300">
              Email address
            </label>
            <input
              id="email"
              type="email"
              placeholder="Email"
              className="mt-2 w-full h-12 px-4 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200"
              onChange={(e) =>
                setAuthData({ ...authData, email: e.target.value })
              }
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
              onChange={(e) =>
                setAuthData({ ...authData, password: e.target.value })
              }
            />
            {errors.password && <p className="text-red-500 mt-1">{errors.password}</p>}
          </div>
          <div className="flex justify-between items-center mb-6">
            <Link href="/forgot-password" className="text-blue-500 hover:underline">
              Forgot password?
            </Link>
            <Link href="/register" className="text-blue-500 hover:underline">
              Don't have an account? Sign Up
            </Link>
          </div>
          <button
            type="button"
            className={`w-full py-3 rounded-md text-white font-semibold ${loading ? 'bg-gray-600' : 'bg-blue-600'} hover:bg-blue-700 transition duration-300`}
            onClick={submitForm}
            disabled={loading}
          >
            {loading ? "Processing..." : "Login"}
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
              height={24}
              width={24}
              alt="Google Icon"
              className="mr-3"
            />
            Sign in with Google
          </button>

          <Link
            href="/magic-link"
            className="w-full py-3 rounded-md border border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 text-gray-900 dark:text-gray-100 font-semibold flex items-center justify-center hover:bg-gray-100 dark:hover:bg-gray-600 transition duration-300"
          >
            <Image
              src="/magic_link.png"
              height={24}
              width={24}
              alt="Google Icon"
              className="mr-3"
            />
            Sign in via Email Link
          </Link>
        </div>
      </div>
    </section>
  );
}
