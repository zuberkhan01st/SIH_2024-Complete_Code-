"use client";
import React, { useState, useEffect } from 'react';
import { motion, useAnimation } from 'framer-motion';

const images = [
  "a.jpg",
  "b.jpg",
  "c.jpg",
];

const carouselVariants = {
  hidden: { opacity: 0, x: 50 },
  visible: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -50 },
};

const AnimatedCarousel = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const controls = useAnimation();

  const handleNext = async () => {
    await controls.start("exit");
    setCurrentIndex((prevIndex) => (prevIndex + 1) % images.length);
    await controls.start("visible");
  };

  const handlePrev = async () => {
    await controls.start("exit");
    setCurrentIndex((prevIndex) => (prevIndex - 1 + images.length) % images.length);
    await controls.start("visible");
  };

  useEffect(() => {
    const interval = setInterval(async () => {
      await controls.start("exit");
      setCurrentIndex((prevIndex) => (prevIndex + 1) % images.length);
      await controls.start("visible");
    }, 3000);

    return () => clearInterval(interval); // Clean up the interval on unmount
  }, [controls]);

  return (
    <section className="relative py-16 bg-gray-100 dark:bg-gray-900 text-center">
      <div className="max-w-4xl mx-auto px-4 relative">
        <h2 className="text-3xl font-bold mb-8 sm:text-4xl text-red-800 dark:text-red-400">
          Our Visits to Post Offices
        </h2>
        <div className="relative overflow-hidden">
          <div className="relative flex items-center">
            {images.map((image, index) => (
              <motion.div
                key={index}
                className={`w-full flex-shrink-0 ${index === currentIndex ? 'block' : 'hidden'}`}
                variants={carouselVariants}
                initial="hidden"
                animate={controls}
                exit="exit"
                transition={{ duration: 0.8, ease: "easeInOut" }}
              >
                <img
                  src={image}
                  alt={`Post Office ${index + 1}`}
                  className="w-full h-full object-cover"
                  style={{ height: '400px', objectFit: 'cover' }}
                />
              </motion.div>
            ))}
          </div>
          <button
            onClick={handlePrev}
            className="absolute top-1/2 left-4 transform -translate-y-1/2 bg-gray-800 text-white p-3 rounded-full shadow-lg hover:bg-gray-700 transition"
          >
            &lt;
          </button>
          <button
            onClick={handleNext}
            className="absolute top-1/2 right-4 transform -translate-y-1/2 bg-gray-800 text-white p-3 rounded-full shadow-lg hover:bg-gray-700 transition"
          >
            &gt;
          </button>
        </div>
      </div>
    </section>
  );
};

export default AnimatedCarousel;
