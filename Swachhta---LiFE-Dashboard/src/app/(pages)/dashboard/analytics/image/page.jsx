'use client';
import React, { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { motion } from 'framer-motion';
import { Bar, Pie, Line, Radar } from 'react-chartjs-2';
import { Chart as ChartJS, Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale, ArcElement, PointElement, LineElement, RadialLinearScale } from 'chart.js';
import { toast, ToastContainer } from 'react-toastify';
import 'chart.js/auto';
import 'react-toastify/dist/ReactToastify.css';

ChartJS.register(
  Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale, 
  ArcElement, PointElement, LineElement, RadialLinearScale
);

function Home() {
  const { data: session } = useSession(); // Get session data
  const [result, setResult] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [currentUserId, setCurrentUserId] = useState(''); // State for current user ID

  useEffect(() => {
    if (session?.user?._id) {
      setCurrentUserId(session.user._id);
      console.log(session.user._id); // Set the current user ID from session
    }
  }, [session]);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUploadClick = async () => {
    if (!selectedFile) {
      alert('Please select an image file first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/detect/image', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);

      // Fetch the processed image from the Flask API
      const processedImageUrl = `http://localhost:5000/download/image?image_path=${data.processed_image_path}`;
      setImageUrl(processedImageUrl);

      // Trigger alert if waste percentage exceeds the threshold
      if (data.percentage_waste > 1) {
        await handleTriggerAlert({
          assignedTo: currentUserId,
          message: 'Waste threshold exceeded',
          area: data.total_waste_area,
          details: `Detected waste area: ${data.total_waste_area}, Image area: ${data.image_area}, Waste percentage: ${data.percentage_waste}%`,
        });
      }
    } catch (error) {
      console.error('Error uploading image:', error);
      toast.error('An error occurred while processing the image.');
    }
  };

  const handleTriggerAlert = async ({ assignedTo, message, area, details }) => {
    try {
      const response = await fetch('/api/alerts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ assignedTo, message, area, details }),
      });

      if (response.ok) {
        // Only show success toast when alert has been sent successfully
        toast.error('You have an alert');
      } else {
        const errorData = await response.json();
        toast.success('You have No Alert');
      }
    } catch (error) {
      console.error('Error sending alert:', error);
      toast.error('An error occurred while sending the alert.');
    }
  };

  // Data for Pie, Line, and Radar charts (use result data here)
  const pieData = {
    labels: ['Waste Area', 'Non-Waste Area'],
    datasets: [
      {
        data: [result?.total_waste_area, result?.image_area - result?.total_waste_area],
        backgroundColor: ['rgba(139, 0, 0, 1)', 'rgba(75, 192, 192, 0.6)'],
        borderWidth: 1,
      },
    ],
  };

  const lineData = {
    labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'], // Mock time data
    datasets: [
      {
        label: 'Waste Percentage',
        data: [result?.percentage_waste, 0.8, 0.6, 1.2], // Use actual time data when available
        fill: false,
        borderColor: 'rgba(75, 192, 192, 1)',
        tension: 0.1,
      },
    ],
  };

  const radarData = {
    labels: ['Waste Area', 'Image Area', 'Total Waste Percentage', 'Non-Waste Area', 'Area Percentage'],
    datasets: [
      {
        label: 'Waste Data',
        data: [
          result?.total_waste_area,
          result?.image_area,
          result?.percentage_waste,
          result?.image_area - result?.total_waste_area,
          result?.percentage_waste * 100,
        ],
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="bg-red-500 text-white flex flex-col items-center p-6"
    >
      <h1 className="text-4xl font-bold mb-6 text-center">Upload Image for Detection</h1>

      <input
        type="file"
        onChange={handleFileChange}
        className="mb-6 p-2 bg-white text-black rounded-lg"
      />
      <button
        onClick={handleUploadClick}
        className="mb-6 px-4 py-2 bg-yellow-500 text-black rounded-lg shadow-md hover:bg-yellow-600 transition duration-300"
      >
        Upload Image
      </button>

      {result && (
        <div className="w-full max-w-6xl">
          <h2 className="text-2xl font-semibold mb-4 text-center">Results:</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-6 mb-6">
            {/* Bar Chart */}
            <div className="bg-white p-4 rounded-lg shadow-lg">
              <h3 className="text-lg font-semibold mb-2 text-center">Bar Chart</h3>
              <Bar
                data={{
                  labels: ['Total Waste Area', 'Image Area', 'Percentage Waste'],
                  datasets: [
                    {
                      label: 'Waste Analysis',
                      data: [
                        result.total_waste_area,
                        result.image_area,
                        result.percentage_waste,
                      ],
                      backgroundColor: 'rgba(75, 192, 192, 0.4)',
                      borderColor: 'rgba(75, 192, 192, 1)',
                      borderWidth: 2,
                    },
                  ],
                }}
              />
            </div>

            {/* Pie Chart */}
            <div className="bg-white p-4 rounded-lg shadow-lg">
              <h3 className="text-lg font-semibold mb-2 text-center">Pie Chart</h3>
              <Pie data={pieData} />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-6 mb-6">
            {/* Line Chart */}
            <div className="bg-white p-4 rounded-lg shadow-lg">
              <h3 className="text-lg font-semibold mb-2 text-center">Line Chart</h3>
              <Line data={lineData} />
            </div>

            {/* Radar Chart */}
            <div className="bg-white p-4 rounded-lg shadow-lg">
              <h3 className="text-lg font-semibold mb-2 text-center">Radar Chart</h3>
              <Radar data={radarData} />
            </div>
          </div>
        </div>
      )}

      {imageUrl && (
        <div className="mt-6">
          <h2 className="text-2xl font-semibold mb-2 text-center">Processed Image:</h2>
          <img
            src={imageUrl}
            alt="Processed"
            className="w-full max-w-2xl mx-auto rounded-lg shadow-lg"
          />
        </div>
      )}

      <ToastContainer />
    </motion.div>
  );
}

export default Home;
