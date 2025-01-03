"use client";
import React, { useState, useEffect } from "react";

export default function IpPostOfficeDetails() {
  const [ipDetails, setIpDetails] = useState(null);
  const [postOffices, setPostOffices] = useState([]);
  const [mapUrl, setMapUrl] = useState("");
  const [timezoneDetails, setTimezoneDetails] = useState({});
  const [searchTerm, setSearchTerm] = useState("");

  // Automatically fetch user info on component mount
  useEffect(() => {
    getUserInfo();
  }, []); // Empty dependency array ensures it runs only once on mount

  const getUserInfo = () => {
    fetch("https://api.ipify.org/?format=json")
      .then((res) => res.json())
      .then((data) => {
        const ipAddress = data.ip;

        fetch(`https://ipinfo.io/${ipAddress}?token=ae5196e3d745a4`)
          .then((res) => res.json())
          .then((data) => {
            const [lat, lon] = data.loc.split(",");
            setIpDetails(data);
            setMapUrl(
              `https://maps.google.com/maps?q=${lat},${lon}&z=15&output=embed`
            );
            fetchTimezone(data.timezone, data.postal);
            fetchPostOffices(data.postal);
          });
      });
  };

  const fetchTimezone = (timezone, pincode) => {
    fetch(`https://api.postalpincode.in/pincode/${pincode}`)
      .then((res) => res.json())
      .then((data) => {
        setTimezoneDetails({
          timezone,
          pincode,
          count: data[0].PostOffice.length,
        });
      });
  };

  const fetchPostOffices = (pincode) => {
    fetch(`https://api.postalpincode.in/pincode/${pincode}`)
      .then((res) => res.json())
      .then((data) => setPostOffices(data[0].PostOffice));
  };

  const filteredPostOffices = postOffices.filter((office) =>
    office.Name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="py-16 bg-gray-100 dark:bg-gray-900 text-center">
      <h2 className="text-4xl text-red-700 font-bold mb-4">Find Post Offices Nearby Me</h2>

      {ipDetails && (
        <div className="mt-8 px-4">
          <div className="flex flex-wrap justify-between mb-6">
            <ul className="w-full sm:w-1/3 mb-4 sm:mb-0">
              <li>Latitude: {ipDetails.loc.split(",")[0]}</li>
              <li>Longitude: {ipDetails.loc.split(",")[1]}</li>
            </ul>
            <ul className="w-full sm:w-1/3 mb-4 sm:mb-0">
              <li>City: {ipDetails.city}</li>
              <li>Region: {ipDetails.region}</li>
            </ul>
            <ul className="w-full sm:w-1/3">
              <li>Organization: {ipDetails.org}</li>
              <li>Timezone: {ipDetails.timezone}</li>
            </ul>
          </div>

          {/* Map */}
          <div className="my-6">
            <iframe
              src={mapUrl}
              width="100%"
              height="400"
              allowFullScreen=""
              loading="lazy"
              className="border-2 border-gray-300 rounded-md"
            ></iframe>
          </div>

          {/* Timezone */}
          <div className="mt-6">
            <h3 className="text-xl">Timezone: {timezoneDetails.timezone}</h3>
            <h3 className="text-xl">
              Date and Time:{" "}
              {new Date().toLocaleString("en-US", {
                timeZone: timezoneDetails.timezone,
              })}
            </h3>
            <h3 className="text-xl">Pincode: {timezoneDetails.pincode}</h3>
            <h3 className="text-xl">Post Offices Count: {timezoneDetails.count}</h3>
          </div>

          {/* Search and Post Office List */}
          <div className="my-6">
            <input
              type="text"
              placeholder="Filter Post Offices"
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full p-3 border border-gray-400 rounded-md"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {filteredPostOffices.map((office) => (
              <ul key={office.Name} className="p-4 border border-gray-400 rounded-md">
                <li>Name: {office.Name}</li>
                <li>Branch Type: {office.BranchType}</li>
                <li>Delivery Status: {office.DeliveryStatus}</li>
                <li>District: {office.District}</li>
                <li>Division: {office.Division}</li>
              </ul>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
