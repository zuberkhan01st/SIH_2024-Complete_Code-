'use client'
import React, { useState } from 'react';

function PostOfficeHeadReports() {
  const [reports, setReports] = useState([
    { id: 1, title: 'Monthly Compliance Report', date: '2024-09-01', type: 'Monthly', status: 'Pending', assignedStaff: 'John Doe' },
    { id: 2, title: 'Incident Report - Waste Disposal', date: '2024-09-02', type: 'Incident', status: 'Resolved', assignedStaff: 'Jane Smith' },
  ]);

  const handleSendReport = (newReport) => {
    setReports([...reports, newReport]);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <main className="flex-1 p-4">
        {/* Reports Overview */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white p-4 rounded shadow">
              <h3 className="text-xl font-semibold mb-2">Total Reports</h3>
              <p>{reports.length}</p>
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h3 className="text-xl font-semibold mb-2">Resolved Reports</h3>
              <p>{reports.filter(report => report.status === 'Resolved').length}</p>
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h3 className="text-xl font-semibold mb-2">Pending Reports</h3>
              <p>{reports.filter(report => report.status === 'Pending').length}</p>
            </div>
          </div>
        </section>

        {/* Generate New Report */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Generate New Report</h2>
          <form className="bg-white p-4 rounded shadow mb-4">
            <div className="mb-4">
              <label className="block mb-2 font-semibold">Title</label>
              <input type="text" className="w-full p-2 border rounded" placeholder="Report Title" />
            </div>
            <div className="mb-4">
              <label className="block mb-2 font-semibold">Description</label>
              <textarea className="w-full p-2 border rounded" placeholder="Report Description"></textarea>
            </div>
            <div className="mb-4">
              <label className="block mb-2 font-semibold">Type</label>
              <select className="w-full p-2 border rounded">
                <option>Monthly</option>
                <option>Incident</option>
                <option>Compliance</option>
              </select>
            </div>
            <div className="mb-4">
              <label className="block mb-2 font-semibold">Date Range</label>
              <input type="date" className="w-full p-2 border rounded" />
            </div>
            <div className="mb-4">
              <label className="block mb-2 font-semibold">Assigned Staff</label>
              <input type="text" className="w-full p-2 border rounded" placeholder="Staff Name" />
            </div>
            <div className="mb-4">
              <label className="block mb-2 font-semibold">Attachments</label>
              <input type="file" className="w-full p-2 border rounded" />
            </div>
            <button type="button" onClick={handleSendReport} className="bg-blue-500 text-white px-4 py-2 rounded">Generate Report</button>
          </form>
        </section>

        {/* Current Reports */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Current Reports</h2>
          <table className="w-full bg-white border border-gray-300 rounded shadow">
            <thead>
              <tr>
                <th className="p-2">Title</th>
                <th className="p-2">Date</th>
                <th className="p-2">Type</th>
                <th className="p-2">Assigned Staff</th>
                <th className="p-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {reports.map(report => (
                <tr key={report.id}>
                  <td className="p-2">{report.title}</td>
                  <td className="p-2">{report.date}</td>
                  <td className="p-2">{report.type}</td>
                  <td className="p-2">{report.assignedStaff}</td>
                  <td className="p-2">{report.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        {/* Report History */}
        <section>
          <h2 className="text-2xl font-bold mb-4">Report History</h2>
          <table className="w-full bg-white border border-gray-300 rounded shadow">
            <thead>
              <tr>
                <th className="p-2">Title</th>
                <th className="p-2">Date</th>
                <th className="p-2">Type</th>
                <th className="p-2">Assigned Staff</th>
                <th className="p-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {reports.filter(report => report.status === 'Resolved').map(report => (
                <tr key={report.id}>
                  <td className="p-2">{report.title}</td>
                  <td className="p-2">{report.date}</td>
                  <td className="p-2">{report.type}</td>
                  <td className="p-2">{report.assignedStaff}</td>
                  <td className="p-2">{report.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </main>
    </div>
  );
}

export default PostOfficeHeadReports;
