"use client";

import { useState, useEffect } from "react";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { Bar } from "react-chartjs-2";
import { Chart as ChartJS, Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale } from "chart.js";
import { motion } from "framer-motion";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Moon, Sun } from "lucide-react";

ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale);

export default function AdminPage() {
  const [userDetails, setUserDetails] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [message, setMessage] = useState("");
  const [userAlertsCount, setUserAlertsCount] = useState([]);
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Fetch user details
  useEffect(() => {
    const fetchUserDetails = async () => {
      try {
        const response = await fetch("/api/auth/get-user-ids");
        const data = await response.json();
        setUserDetails(data.userDetails);
      } catch (error) {
        console.error("Failed to fetch user details", error);
        toast.error("Failed to fetch user details");
      }
    };

    const fetchUserAlertsCount = async () => {
      try {
        const response = await fetch("/api/counts");
        const data = await response.json();
        if (data.success) {
          setUserAlertsCount(data.userAlertsCount);
        } else {
          toast.error("Failed to fetch user alerts count");
        }
      } catch (error) {
        console.error("Error fetching user alerts count:", error);
        toast.error("An error occurred while fetching data.");
      }
    };

    fetchUserDetails();
    fetchUserAlertsCount();

    // Set initial dark mode based on system preference
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      setIsDarkMode(true);
    }
  }, []);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  const handleSendMessage = async () => {
    try {
      const response = await fetch("/api/auth/send-message", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ userId: selectedUserId, message }),
      });
      const result = await response.json();

      if (response.ok) {
        toast.success("Message sent successfully!");
      } else {
        toast.error(result.error || "Failed to send message");
      }
    } catch (error) {
      console.error("Failed to send message", error);
      toast.error("An error occurred");
    }
  };

  const chartData = {
    labels: userAlertsCount.map((user) => user.name),
    datasets: [
      {
        label: "No. of Alerts",
        data: userAlertsCount.map((user) => user.alertCount),
        backgroundColor: isDarkMode ? "rgba(255, 159, 64, 0.6)" : "rgba(75, 192, 192, 0.6)",
        borderColor: isDarkMode ? "rgba(255, 159, 64, 1)" : "rgba(75, 192, 192, 1)",
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: "Alerts Per User",
        color: isDarkMode ? "#fff" : "#000",
      },
      tooltip: {
        backgroundColor: isDarkMode ? "#333" : "#fff",
        titleColor: isDarkMode ? "#fff" : "#000",
      },
    },
    scales: {
      x: {
        ticks: {
          color: isDarkMode ? "#fff" : "#000",
        },
      },
      y: {
        ticks: {
          color: isDarkMode ? "#fff" : "#000",
        },
      },
    },
  };

  return (
    <div className={`p-6 ${isDarkMode ? "bg-gray-900 text-white" : "bg-white text-black"} min-h-screen`}>

      <div className="mb-4 text-center flex justify-between items-center">
      <h1 className="text-2xl font-bold text-center mb-6">Admin Page</h1>
        <button onClick={toggleDarkMode} className="px-4 py-2 bg-gray-700 text-white rounded-md">
           {isDarkMode ? <Sun/> : <Moon/>} 
        </button>
      </div>

      {/* User Alerts Summary */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">User Alerts Summary</h2>
        <table className="min-w-full table-auto border-collapse shadow-lg rounded-lg mb-6">
          <thead>
            <tr className={`${isDarkMode ? "bg-gray-800" : "bg-gray-200"}`}>
              <th className="p-4 text-left">User Name</th>
              <th className="p-4 text-left">No. of Alerts</th>
            </tr>
          </thead>
          <tbody>
            {userAlertsCount.length === 0 ? (
              <tr>
                <td colSpan="2" className="p-4 text-center">No data available</td>
              </tr>
            ) : (
              userAlertsCount.map((user) => (
                <tr key={user.userId} className="border-b">
                  <td className="p-4">{user.name}</td>
                  <td className="p-4">{user.alertCount}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
        {userAlertsCount.length > 0 && (
          <div className="chart-container">
            <Bar data={chartData} options={chartOptions} />
          </div>
        )}
      </div>

      {/* Send Message */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Send Message to User</h2>
        <motion.select
          id="userSelect"
          value={selectedUserId}
          onChange={(e) => setSelectedUserId(e.target.value)}
          className="border border-gray-300 dark:border-gray-700 p-2 rounded w-full bg-white dark:bg-gray-800 text-black dark:text-white"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <option value="">Select User</option>
          {userDetails.map(({ id, name }) => (
            <option key={id} value={id}>
              {name}
            </option>
          ))}
        </motion.select>

        {selectedUserId && (
          <div className="mt-4">
            <label htmlFor="message" className="block font-semibold mb-2">
              Message:
            </label>
            <textarea
              id="message"
              placeholder="Type your message here"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              className="border border-gray-300 dark:border-gray-700 p-2 rounded w-full bg-white dark:bg-gray-800 text-black dark:text-white"
            />
            <motion.button
              onClick={handleSendMessage}
              className="bg-blue-500 hover:bg-blue-600 dark:bg-blue-400 dark:hover:bg-blue-500 text-white px-4 py-2 rounded mt-2"
              disabled={!message}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Send Message
            </motion.button>
          </div>
        )}
      </div>

      <Link href="/admin/dashboard">
        <Button className="bg-red-500 hover:bg-red-600 dark:bg-red-400 dark:hover:bg-red-500 text-white font-bold py-3 px-6 rounded-lg shadow-lg transition-transform transform hover:scale-105">
          Back
        </Button>
      </Link>

      <ToastContainer />
    </div>
  );
}
