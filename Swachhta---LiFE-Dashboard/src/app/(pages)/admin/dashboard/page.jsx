"use client"
import { Button } from "@/components/ui/button";
import Link from "next/link";
import React from "react";
import { motion } from "framer-motion";

export default function AdminDashboard() {
  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 flex flex-col">
      {/* Hero Section */}
      <motion.div
        className="relative h-screen bg-gradient-to-br from-blue-500 via-blue-700 to-blue-900 dark:from-gray-800 dark:via-gray-900 dark:to-black flex flex-col justify-center items-center text-center p-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        <h1 className="text-5xl font-extrabold text-white mb-6">
          Welcome, Divisional Officer
        </h1>
        <p className="text-lg text-blue-100 dark:text-gray-400 max-w-2xl">
          Seamlessly manage Swachhta and LiFE practices across all post offices in your division with an intuitive and powerful dashboard.
        </p>
        <div className="mt-8 flex gap-4">
          <Link href="/admin/dashboard/allUsers">
            <Button className="bg-yellow-400 hover:bg-yellow-500 text-black font-bold py-3 px-6 rounded-lg shadow-lg">
              View Post Masters Details
            </Button>
          </Link>
          <Link href="/admin/dashboard/life-practice">
            <Button className="bg-green-400 hover:bg-green-500 text-black font-bold py-3 px-6 rounded-lg shadow-lg">
              Life Practice
            </Button>
          </Link>
        </div>
        <motion.div
          className="absolute bottom-8 text-white text-sm animate-bounce"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2, duration: 1 }}
        >
          For Divisional Officers Only
        </motion.div>
      </motion.div>
    </div>
  );
}
