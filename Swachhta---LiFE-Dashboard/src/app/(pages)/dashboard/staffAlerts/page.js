'use client';
import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { motion } from 'framer-motion';

const Alerts = () => {
  const { data: session, status } = useSession(); // Get session data
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAlerts = async () => {
      if (session?.user?._id) {
        const url = `/api/alerts?assignedTo=${session.user._id}`; // Get alerts for the logged-in user
        try {
          const res = await fetch(url);
          const data = await res.json();

          if (data.success) {
            setAlerts(data.alerts); // Set alerts fetched from the database
          } else {
            console.error('Error fetching alerts:', data.error);
          }
        } catch (error) {
          console.error('Error fetching alerts:', error);
        } finally {
          setLoading(false);
        }
      }
    };

    if (status === 'authenticated') {
      fetchAlerts();
    }
  }, [session, status]);

  if (status === 'loading' || loading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
        className="flex justify-center items-center min-h-screen"
      >
        <p>Loading...</p>
      </motion.div>
    );
  }

  if (status === 'unauthenticated') {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
        className="flex justify-center items-center min-h-screen"
      >
        <p>Please log in to view alerts.</p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
      className="p-8 bg-gray-50 dark:bg-gray-900 min-h-screen"
    >
      <motion.h1
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 1 }}
        className="text-3xl font-bold mb-8 text-center text-indigo-700 dark:text-indigo-400"
      >
        My Alerts
      </motion.h1>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 1 }}
        className="overflow-x-auto rounded-lg shadow-lg"
      >
        {alerts.length > 0 ? (
          <motion.table
            className="min-w-full bg-white dark:bg-gray-800 shadow-md rounded-lg border border-gray-300 dark:border-gray-700"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 1 }}
          >
            <thead className="bg-indigo-600 text-white dark:bg-indigo-500">
              <tr>
                <th className="py-3 px-4 text-left">Message</th>
                <th className="py-3 px-4 text-left">Area</th>
                <th className="py-3 px-4 text-left">Details</th>
                <th className="py-3 px-4 text-left">Status</th>
                <th className="py-3 px-4 text-left">Assigned To</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((alert) => (
                <motion.tr
                  key={alert._id}
                  className="border-b hover:bg-indigo-50 dark:hover:bg-indigo-900"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5, duration: 1 }}
                >
                  <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                    {alert.message}
                  </td>
                  <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                    {alert.area}
                  </td>
                  <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                    {alert.details || 'No details provided'}
                  </td>
                  <td className="py-3 px-4">
                    <span
                      className={`px-3 py-1 rounded-full text-white ${
                        alert.status === 'Pending'
                          ? 'bg-yellow-500 dark:bg-yellow-600'
                          : alert.status === 'In Progress'
                          ? 'bg-blue-500 dark:bg-blue-600'
                          : 'bg-green-500 dark:bg-green-600'
                      }`}
                    >
                      {alert.status}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                    {alert.assignedTo?.name || 'Unassigned'}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </motion.table>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 1 }}
            className="text-center text-gray-600 dark:text-gray-400 py-10"
          >
            <p>No alerts assigned to you.</p>
          </motion.div>
        )}
      </motion.div>
    </motion.div>
  );
};

export default Alerts;
