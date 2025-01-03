"use client";
import React from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function ActionsSection() {
  return (
    <motion.section
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
      className="p-4 md:p-6 flex flex-col gap-4 md:gap-6"
    >
      <h2 className="text-xl md:text-2xl font-semibold mb-2 md:mb-4">Actions</h2>
      
      {/* Section 1: Buttons */}
      <div className="flex flex-col md:flex-row gap-4 md:gap-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <Link href='/dashboard/staffReports'>
          <Button className="bg-blue-600 text-white p-4 rounded-lg shadow-md flex-1 transition-transform transform hover:scale-105 dark:bg-blue-700">
            Generate Report
          </Button>
          </Link>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Link href='/dashboard/staffAlerts'>
          <Button className="bg-blue-700 text-white p-4 rounded-lg shadow-md flex-1 transition-transform transform hover:scale-105 dark:bg-blue-800">
            Review Alerts
          </Button>
          </Link>
        </motion.div>
      </div>

      {/* Section 2: Additional Actions */}
      <div className="mt-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="flex flex-col md:flex-row gap-4 md:gap-6"
        >
          <div className="flex-1">
            <Link href={"/dashboard/postHead"}>
            <Button className="bg-green-600 text-white p-4 rounded-lg shadow-md w-full transition-transform transform hover:scale-105 dark:bg-green-700">
              View Reports
            </Button>
            </Link>
          </div>
          <div className="flex-1">
            <Button className="bg-yellow-600 text-white p-4 rounded-lg shadow-md w-full transition-transform transform hover:scale-105 dark:bg-yellow-700">
              Update Settings
            </Button>
          </div>
        </motion.div>
      </div>

      {/* Section 3: Additional Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
        className="mt-6 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg shadow-md"
      >
        <h3 className="text-lg font-semibold mb-2">Additional Information</h3>
        <p>
          Use these buttons to manage your dashboard efficiently. Each action is designed to improve your workflow and provide quick access to essential features.
        </p>
      </motion.div>
    </motion.section>
  );
}
