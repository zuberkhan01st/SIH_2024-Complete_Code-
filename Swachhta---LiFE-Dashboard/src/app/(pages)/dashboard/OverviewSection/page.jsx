"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export default function LifePracticesWeather() {
  const [weatherData, setWeatherData] = useState(null);

  useEffect(() => {
    async function fetchWeather() {
      try {
        const response = await fetch("/api/weather"); // Your API endpoint for weather data
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setWeatherData(data);
      } catch (error) {
        console.error("Error fetching weather data:", error);
      }
    }
    fetchWeather();
  }, []);

  if (!weatherData) return <p className="text-center">Loading weather data...</p>;

  const { city, list } = weatherData.forecast;

  return (
    <div className="min-h-screen transition-colors bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <header className="p-6 flex justify-between items-center bg-green-500 dark:bg-green-700">
        <h1 className="text-2xl font-bold">Weather for Your Health - {city.name}</h1>
      </header>

      <motion.div
        className="p-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {list.slice(0, 5).map((item, index) => (
          <motion.div
            key={index}
            className="bg-green-100 dark:bg-green-800 rounded-lg shadow p-4"
            whileHover={{ scale: 1.05 }}
          >
            <h2 className="text-lg font-semibold">
              {new Date(item.dt * 1000).toLocaleString()}
            </h2>
            <p className="text-sm">Temperature: {item.main.temp}°C</p>
            <p className="text-sm">Feels Like: {item.main.feels_like}°C</p>
            <p className="text-sm capitalize">Condition: {item.weather[0].description}</p>
            <p className="text-sm">Wind: {item.wind.speed} m/s</p>
            <p className="text-sm mt-4">Wellness Tip: {getWellnessTip(item)}</p>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}

// Helper function to provide wellness tips based on weather
const getWellnessTip = (weather) => {
  const temp = weather.main.temp;
  if (temp > 30) {
    return "Stay hydrated and avoid too much sun!";
  } else if (temp < 15) {
    return "Dress warmly and consider indoor activities.";
  } else {
    return "Perfect weather for a walk or outdoor exercise!";
  }
};
