"use client";
import React from "react";
import { motion } from "framer-motion";

export default function DashboardLayout({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen flex flex-col"
    >
      <main className="flex-1 p-4 md:p-6 bg-gray-100 dark:bg-gray-900">
        {children}
      </main>
    </motion.div>
  );
}
