"use client";

import React, { useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { useSearchParams } from "next/navigation";
import Link from "next/link";

export default function ResetPassword({ params }) {
  const searchParam = useSearchParams();
  const [authState, setAuthState] = useState({
    password: "",
    cpassword: "",
  });
  const [loading, setLoading] = useState(false);

  const submit = (event) => {
    event.preventDefault();
    setLoading(true);
    axios
      .post("/api/auth/reset-password", {
        email: params.email,
        signature: searchParam.get("signature"),
        password: authState.password,
        password_confirmation: authState.cpassword,
      })
      .then((res) => {
        setLoading(false);
        const response = res.data;
        if (response.status === 400) {
          toast.error(response.message, { theme: "colored" });
        } else if (response.status === 200) {
          toast.success(response.message, { theme: "colored" });
        }
      })
      .catch((err) => {
        setLoading(false);
        console.log("Error:", err);
      });
  };

  return (
    <>
      <ToastContainer />
      <div className="h-screen w-screen flex justify-center items-center bg-gray-100 dark:bg-gray-900">
        <div className="w-full max-w-md p-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Reset Password
          </h1>
          <form onSubmit={submit}>
            <div className="mt-5">
              <label className="block text-gray-700 dark:text-gray-300">Password</label>
              <input
                type="password"
                placeholder="Enter your new password"
                className="w-full h-12 p-3 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
                onChange={(event) =>
                  setAuthState({ ...authState, password: event.target.value })
                }
              />
            </div>
            <div className="mt-5">
              <label className="block text-gray-700 dark:text-gray-300">Confirm Password</label>
              <input
                type="password"
                placeholder="Confirm your new password"
                className="w-full h-12 p-3 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
                onChange={(event) =>
                  setAuthState({ ...authState, cpassword: event.target.value })
                }
              />
            </div>
            <div className="mt-5">
              <button
                type="submit"
                className={`w-full h-12 bg-black text-white rounded-md ${loading ? 'bg-gray-600' : 'hover:bg-gray-700'} transition duration-300`}
                disabled={loading}
              >
                {loading ? "Processing..." : "Submit"}
              </button>
            </div>
            <div className="mt-5 text-center">
              <Link href="/login" className="text-blue-500 hover:underline">
                Back to Login
              </Link>
            </div>
          </form>
        </div>
      </div>
    </>
  );
}
