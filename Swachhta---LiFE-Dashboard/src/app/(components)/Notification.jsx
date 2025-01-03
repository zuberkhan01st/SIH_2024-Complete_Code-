'use client';
import { useEffect, useState } from "react";
import { useSession } from "next-auth/react"; // For session management
import { Bell } from "lucide-react"; // Notification icon from Lucide
import { motion, AnimatePresence } from "framer-motion"; // For smooth animations
import Link from "next/link";

const Notification = () => {
  const { data: session, status } = useSession(); // Get session data
  const [alerts, setAlerts] = useState([]);
  const [showDialog, setShowDialog] = useState(false);

  // Fetch alerts for the logged-in users
  useEffect(() => {
    const fetchAlerts = async () => {
      if (session?.user?._id) {
        const url = `/api/alerts?assignedTo=${session.user._id}`; // Fetch alerts for the user
        try {
          const res = await fetch(url);
          const data = await res.json();
          if (data.success) {
            const userAlerts = data.alerts.filter(
              (alert) => alert.assignedTo?._id === session.user._id
            );
            setAlerts(userAlerts);
          } else {
            console.error("Error fetching alerts:", data.error);
          }
        } catch (error) {
          console.error("Error fetching alerts:", error);
        }
      }
    };

    if (status === "authenticated") {
      fetchAlerts();
    }
  }, [session, status]);

  return (
    <div className="relative">
      {/* Bell Icon with Notification Count */}
      <div
        className="relative cursor-pointer"
        onClick={() => setShowDialog(!showDialog)} // Toggle dialog visibility
      >
        <Bell className="w-8 h-8 text-black" />
        {alerts.length > 0 && (
          <motion.div
            className="absolute top-0 right-0 bg-blue-600 text-white text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
          >
            {alerts.length}
          </motion.div>
        )}
      </div>

      {/* Notification Dialog */}
      <AnimatePresence>
        {showDialog && (
          <motion.div
            className="absolute top-12 right-0 w-72 bg-white shadow-lg rounded-lg p-4 z-10"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <h3 className="text-lg font-bold text-indigo-600 mb-2">
              Notifications
            </h3>
            <Link href={"/dashboard/staffAlerts"}>
            <div className="max-h-60 overflow-y-auto">
              {alerts.length > 0 ? (
                alerts.map((alert, index) => (
                  <div
                    key={index}
                    className="text-sm text-gray-800 border-b py-2 last:border-none"
                  >
                    {alert.message}
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500">
                  No new notifications.
                </p>
              )}
            </div>
            </Link>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Notification;
