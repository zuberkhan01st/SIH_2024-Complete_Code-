"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import Sidebar from "../../(components)/Sidebar";
import Header from "../../(components)/Header";
import Footer from "../../(components)/Footer";

export default function DashboardLayout({ children }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen flex flex-col md:flex-row"
    >
      {/* Sidebar */}
      <Sidebar
        isOpen={isSidebarOpen}
        toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
      />

      {/* Main Content */}
      <div
        className={`flex-1 flex flex-col transition-all duration-300 ease-in-out ${
          isSidebarOpen ? "md:ml-64 ml-0" : "md:ml-0"
        }`}
      >
        {/* Header */}
        <Header
          toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        />

        {/* Main Content Area */}
        <main className="flex-1 p-4 md:p-6 md:ml-64">
          {children}
        </main>

        {/* Footer */}
        <Footer />
      </div>
    </motion.div>
  );
}
