'use client'
import React, { useState } from 'react';

function NonComplianceReviewPage() {
  const [selectedReport, setSelectedReport] = useState(null);
  const reports = [
    {
      id: 1,
      date: '2024-09-01',
      location: 'Main Office',
      status: 'Pending',
      description: 'Trash not disposed of properly.',
      photos: ['photo1.jpg', 'photo2.jpg'],
      videos: ['video1.mp4'],
    },
    // Add more reports as needed
  ];

  const handleSelectReport = (report) => {
    setSelectedReport(report);
  };

  const handleCreateAlert = () => {
    // Function to create alert
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <main className="flex-1 p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <section className="col-span-1">
            <h2 className="text-xl font-bold mb-4">Reports</h2>
            <ul className="bg-white p-4 rounded shadow">
              {reports.map((report) => (
                <li
                  key={report.id}
                  className={`p-2 mb-2 dark:text-black cursor-pointer rounded ${selectedReport?.id === report.id ? 'bg-blue-100' : 'bg-gray-100'}`}
                  onClick={() => handleSelectReport(report)}
                >
                  <p>Report ID: {report.id}</p>
                  <p>Date: {report.date}</p>
                  <p>Location: {report.location}</p>
                  <p>Status: {report.status}</p>
                </li>
              ))}
            </ul>
          </section>

          {selectedReport && (
            <section className="col-span-2">
              <h2 className="text-xl font-bold mb-4">Report Details</h2>
              <div className="bg-white p-4 rounded shadow mb-4">
                <p><strong>Date:</strong> {selectedReport.date}</p>
                <p><strong>Location:</strong> {selectedReport.location}</p>
                <p><strong>Description:</strong> {selectedReport.description}</p>
                <div>
                  <strong>Photos:</strong>
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    {selectedReport.photos.map((photo, index) => (
                      <img key={index} src={photo} alt={`Report Photo ${index + 1}`} className="w-full h-32 object-cover rounded" />
                    ))}
                  </div>
                </div>
                <div className="mt-4">
                  <strong>Videos:</strong>
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    {selectedReport.videos.map((video, index) => (
                      <video key={index} controls className="w-full h-32 object-cover rounded">
                        <source src={video} type="video/mp4" />
                        Your browser does not support the video tag.
                      </video>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-white p-4 rounded shadow mb-4">
                <h3 className="text-lg font-bold mb-2">Create Alert</h3>
                <form onSubmit={handleCreateAlert}>
                  <div className="mb-4">
                    <label className="block mb-2">Title</label>
                    <input type="text" className="w-full p-2 border border-gray-300 rounded" required />
                  </div>
                  <div className="mb-4">
                    <label className="block mb-2">Description</label>
                    <textarea className="w-full p-2 border border-gray-300 rounded" rows="4" required></textarea>
                  </div>
                  <div className="mb-4">
                    <label className="block mb-2">Priority</label>
                    <select className="w-full p-2 border border-gray-300 rounded" required>
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                  </div>
                  <div className="mb-4">
                    <label className="block mb-2">Due Date</label>
                    <input type="date" className="w-full p-2 border border-gray-300 rounded" required />
                  </div>
                  <button type="submit" className="bg-blue-500 text-white px-4 py-2 rounded">Create Alert</button>
                </form>
              </div>

              <div className="bg-white p-4 rounded shadow">
                <h3 className="text-lg font-bold mb-2">Comments</h3>
                <textarea className="w-full p-2 border border-gray-300 rounded mb-4" rows="4" placeholder="Add your comments here..."></textarea>
                <button className="bg-green-500 text-white px-4 py-2 rounded">Add Comment</button>
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

export default NonComplianceReviewPage;
