"use client";
import React from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { ModeToggle } from "@/components/ui/modeToggle";
import CursorTrail from "@/components/CursorTrail";
import IpPostOfficeDetails from "@/app/(components)/IpPostOfficeDetails";
import AnimatedCarousel from "@/app/(components)/AnimatedCarousel";
// import SplineModel from "@/app/(components)/SplineModel";

export default function LandingPage() {
  return (
    <div className="relative min-h-screen flex flex-col bg-gradient-to-b from-white to-gray-200 dark:from-gray-900 dark:to-black text-gray-900 dark:text-gray-100 overflow-hidden scroll-smooth">
      {/* Cursor Trail */}
      <CursorTrail color="#f56565"/>

      {/* Header */}
      <header className="w-full flex justify-between items-center px-6 py-4 bg-red-600 dark:bg-red-800 text-white shadow-md">
        <div className="flex items-center">
          <img
            src="/post-logo.png"
            alt="Post Office Logo"
            className="w-24 h-12"
          />
        </div>
        <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center">
          Swacch Bharat & LiFE Practices 
        </h1>
        <div className="flex items-center space-x-2 sm:space-x-4">
          <ModeToggle />
          <Link href="/login">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-3 py-1 sm:px-4 sm:py-2 bg-red-700 dark:bg-red-500 text-white rounded-md hover:bg-red-800 dark:hover:bg-red-600 transition"
            >
              Login
            </motion.button>
          </Link>
          <Link href="/admin">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-3 py-1 sm:px-4 sm:py-2 bg-red-700 dark:bg-red-500 text-white rounded-md hover:bg-red-800 dark:hover:bg-red-600 transition"
            >
              Admin
            </motion.button>
          </Link>
        </div>
      </header>

      {/* Navbar */}
      <nav className="w-full bg-gray-700 dark:bg-gray-800 text-white shadow-md">
        <ul className="flex justify-center space-x-4 text-sm sm:text-base py-2">
          <li>
            <Link href="#home" className="hover:text-gray-200">
              Home
            </Link>
          </li>
          <li>
            <Link href="#about" className="hover:text-gray-200">
              About
            </Link>
          </li>
          <li>
            <Link href="#features" className="hover:text-gray-200">
              Features
            </Link>
          </li>
          <li>
            <Link href="#visits" className="hover:text-gray-200">
              Visits
            </Link>
          </li>
          <li>
            <Link href="#postoffices" className="hover:text-gray-200">
              Post Offices
            </Link>
          </li>
          <li>
            <Link href="#solution" className="hover:text-gray-200">
              Our Solution
            </Link>
          </li>
        </ul>
      </nav>

      {/* Introduction Section */}
      <section id="home" className="w-full flex flex-col items-center justify-center bg-gray-100 dark:bg-gray-900 py-16 px-6">
        <div className="flex flex-col lg:flex-row items-center justify-between w-full max-w-6xl mx-auto gap-5">
          {/* Image */}
          <motion.div
            className="w-full lg:w-2/5 mb-8 lg:mb-0"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 1.2 }}
          >
            <img
              src="/postoffice.jpg"
              alt="Post Office Cleanliness"
              className="w-full h-full object-cover rounded-lg shadow-lg"
            />
          </motion.div>

          <motion.div
            className="w-full lg:w-1/2 text-left"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 1.4 }}
          >
            <h1 className="text-4xl font-extrabold sm:text-5xl text-red-800 dark:text-red-400 mb-4">
              Introduction
            </h1>
            <p className="text-lg text-gray-700 dark:text-gray-300 mb-4">
              The Department of Posts is committed to institutionalizing Swachhta protocols and adopting Lifestyle for Environment (LiFE) practices across its network. This solution utilizes AI and pictorial data-based automated monitoring to achieve these goals.
            </p>
            <div className="mt-8">
              <Link href="#about">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-6 py-3 bg-transparent border-2 border-red-700 text-red-700 rounded-md hover:bg-red-700 hover:text-white transition"
                >
                  Learn More
                </motion.button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-16 bg-gray-100 dark:bg-gray-900 px-6">
        <motion.div
          className="max-w-5xl mx-auto"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2 }}
        >
          <h2 className="text-3xl font-bold mb-8 sm:text-4xl text-red-800 dark:text-red-400">
            About the Dashboard
          </h2>
          <p className="text-lg text-gray-700 dark:text-gray-300 mb-4">
            <b>Swachhta Focus:</b> Our system continuously monitors post offices using real-time image processing to maintain top cleanliness standards. It quickly identifies areas needing attention, ensuring a consistently clean environment.
          </p>
          <p className="text-lg text-gray-700 dark:text-gray-300">
            <b>LiFE (Lifestyle for Environment):</b> We promote eco-friendly practices by monitoring waste management and encouraging sustainable behaviors.
          </p>
        </motion.div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-16 bg-gray-100 dark:bg-gray-900 px-6">
        <motion.div
          className="max-w-5xl mx-auto"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2 }}
        >
          <h2 className="text-3xl font-bold mb-8 sm:text-4xl text-red-900 dark:text-red-600">
            Key Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              {
                title: "AI Monitoring",
                description:
                  "Advanced AI technology to continuously monitor cleanliness and green practices across the nation.",
              },
              {
                title: "Real-time Alerts",
                description:
                  "Instant alerts for deviations from standards to ensure timely intervention and corrective actions.",
              },
              {
                title: "Dashboard Features",
                description:
                  "Interactive Data Visualization, Monthly and Yearly Reports, and Two-Tier Login System.",
              },
            ].map((feature, index) => (
              <motion.div
                key={index}
                className="w-full p-6 bg-gray-100 dark:bg-gray-700 rounded-lg shadow-lg"
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8 }}
              >
                <h3 className="text-2xl font-semibold mb-4 text-red-900 dark:text-red-500">
                  {feature.title}
                </h3>
                <p className="text-lg text-gray-700 dark:text-gray-300">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      <section id="visits" className="py-16 bg-gray-100 dark:bg-gray-900 px-6">
        <motion.div
          className="max-w-5xl mx-auto"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2 }}
        >
      <AnimatedCarousel />
        </motion.div>
      </section>

      <section id="postoffices" className="py-16 bg-gray-100 dark:bg-gray-900 px-6">
        <motion.div
          className="max-w-5xl mx-auto"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2 }}
        >
      <IpPostOfficeDetails />
        </motion.div>
      </section>

      {/* <section id="solution" className="py-16 bg-gray-100 dark:bg-gray-900 px-6">
        <motion.div
          className="max-w-5xl mx-auto my-5"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2 }}
        >
           <div>
          <h2  className="text-3xl font-bold mb-8 sm:text-4xl text-red-800 dark:text-red-400">
            Our Solution : 3D Visualization of Post Offices Waste Management
          </h2>
          <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
            This 3D model showcases the waste management process in various post offices.
            It highlights areas where waste is detected and provides insight into how
            eco-friendly practices are being implemented. The model is interactive,
            allowing users to explore different perspectives and analyze the data.
          </p>
        </div>
        <ul className="w-full list-disc pl-6 text-gray-700 dark:text-gray-300 space-y-2">
          <li>Real-time waste detection in post offices.</li>
          <li>Shows waste area and percentage with 3D visualizations.</li>
          <li>Helps monitor eco-friendly initiatives in real-time.</li>
        </ul>
        </motion.div>
      </section> */}

      {/* <section className="cursor-none"><SplineModel/></section> */}

      {/* Footer */}
      <footer className="bg-red-600 text-white">
        <motion.div
          className="container mx-auto text-center"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
            <h3 className="text-2xl font-semibold text-white">
              Swachh Bharat Initiative
            </h3>
            <p className="text-gray-700 dark:text-gray-300 text-sm">
            © {new Date().getFullYear()} Swachh Bharat. All rights reserved.
          </p>
        </motion.div>
      </footer>
    </div>
  );
}
