"use client";
import React, { useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

export default function MagicLink() {
  const [email, setEmail] = useState("");
  const [errors, setErrors] = useState({ email: "" });
  const [loading, setLoading] = useState(false);

  const handleSubmit = (event) => {
    event.preventDefault();
    setLoading(true);
    axios
      .post("/api/auth/magic-link", { email })
      .then((res) => {
        setLoading(false);
        const response = res.data;
        if (response.status === 400) {
          setErrors(response.errors);
        } else if (response.status === 200) {
          toast.success(response.message, { theme: "colored" });
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
            Magic Link
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Enter your email address and we'll send you a magic link to sign in.
          </p>
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <input
                type="email"
                className="w-full h-12 px-4 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200"
                placeholder="Enter your email"
                onChange={(e) => setEmail(e.target.value)}
              />
              {errors.email && <p className="text-red-500 mt-1">{errors.email}</p>}
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
