"use client";
import React, { useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

export default function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState(null);

  const submit = (event) => {
    event.preventDefault();
    setLoading(true);
    axios
      .post("/api/auth/forgot-password", { email: email })
      .then((res) => {
        setLoading(false);
        const response = res.data;
        if (response.status === 200) {
          toast.success(response.message, { theme: "colored" });
        } else if (response.status === 400) {
          setErrors(response.errors);
        } else if (response.status === 500) {
          toast.error(response.message, { theme: "colored" });
        }
      })
      .catch((err) => {
        setLoading(false);
        console.log("The error is", err);
      });
  };

  return (
    <>
      <ToastContainer />
      <section className="bg-gray-50 dark:bg-gray-900 min-h-screen flex items-center justify-center">
        <div className="w-full max-w-md p-8 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Forgot Password?
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Don't worry, it happens. Just enter your email below, and we will send
            an email to you.
          </p>
          <form onSubmit={submit} className="space-y-5">
            <div>
              <label htmlFor="email" className="block text-gray-700 dark:text-gray-300">
                Email
              </label>
              <input
                id="email"
                type="email"
                placeholder="ayush@gmail.com"
                className="mt-2 w-full h-12 px-4 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200"
                onChange={(event) => setEmail(event.target.value)}
              />
              {errors?.email && <p className="text-red-500 mt-1">{errors.email}</p>}
            </div>
            <div>
              <button
                type="submit"
                className={`w-full py-3 rounded-md text-white font-semibold ${loading ? 'bg-gray-600' : 'bg-blue-600'} hover:bg-blue-700 transition duration-300`}
                disabled={loading}
              >
                {loading ? "Processing..." : "Submit"}
              </button>
            </div>
          </form>
        </div>
      </section>
    </>
  );
}
