"use client";

import { useState, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import Chart from 'chart.js/auto';

const THRESHOLD = {
    "Alerts Resolved": 5000,
    "Greenery Detection": 200,
    "Sustainability Promoted": 150,
    "Energy Efficiency Improved": 100,
};

export default function PostOfficeStats() {
    const [data, setData] = useState([]);
    const [filteredData, setFilteredData] = useState([]);
    const [level, setLevel] = useState('State'); // Tracks hierarchy level: State, Divisional, City
    const [selected, setSelected] = useState(null); // Tracks selected item (state or divisional)

    useEffect(() => {
        // Load JSON data from the public folder
        fetch('/post_office_stats.json')
            .then((response) => response.json())
            .then((jsonData) => setData(jsonData));
    }, []);

    useEffect(() => {
        if (level === 'State') {
            setFilteredData(
                data.reduce((acc, item) => {
                    const existing = acc.find((d) => d.State === item.State);
                    if (!existing) {
                        acc.push({
                            State: item.State,
                            "Alerts Resolved": 0,
                            "Greenery Detection": 0,
                            "Sustainability Promoted": 0,
                            "Energy Efficiency Improved": 0,
                        });
                    }
                    const state = acc.find((d) => d.State === item.State);
                    state["Alerts Resolved"] += item["Alerts Resolved"];
                    state["Greenery Detection"] += item["Greenery Detection"];
                    state["Sustainability Promoted"] += item["Sustainability Promoted"];
                    state["Energy Efficiency Improved"] += item["Energy Efficiency Improved"];
                    return acc;
                }, [])
            );
        } else if (level === 'Divisional' && selected) {
            setFilteredData(data.filter((item) => item.State === selected));
        } else if (level === 'City' && selected) {
            setFilteredData(data.filter((item) => item.Divisional === selected));
        }
    }, [level, selected, data]);

    const handleClick = (item) => {
        if (level === 'State') {
            setLevel('Divisional');
            setSelected(item.State);
        } else if (level === 'Divisional') {
            setLevel('City');
            setSelected(item.Divisional);
        }
    };

    const calculateAverage = (item) => {
        const total = item["Alerts Resolved"] + item["Greenery Detection"] + item["Sustainability Promoted"] + item["Energy Efficiency Improved"];
        return total / 4; // Average of the four metrics
    };

    const generateChartData = () => {
        const labels = filteredData.map((item) => (level === 'State' ? item.State : level === 'Divisional' ? item.Divisional : item.City));
        const averages = filteredData.map((item) => calculateAverage(item));

        // Find the minimum and maximum averages
        const minAverage = Math.min(...averages);
        const maxAverage = Math.max(...averages);

        const getColor = (value) => {
            // Color scale: Red for lowest, Green for highest, others are in between
            if (value === minAverage) {
                return 'rgba(255, 99, 132, 1)'; // Red for lowest (poor)
            }
            if (value === maxAverage) {
                return 'rgba(0, 128, 0, 1)'; // Green for highest (best)
            }
            // Other values between
            const ratio = (value - minAverage) / (maxAverage - minAverage);
            const red = Math.round(255 * (1 - ratio));
            const green = Math.round(192 * ratio);
            return 'rgba(`${red}`, `${green}`, 132, 0.6)'; // Gradients between red and green (average)
        };

        return {
            labels,
            datasets: [
                {
                    label: 'Average of 4 Parameters (Alerts Resolved, Greenery Detection, Sustainability Promoted, and Energy Efficiency Improved)',
                    data: averages,
                    backgroundColor: averages.map((value) => getColor(value)),
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                },
            ],
        };
    };

    const options = {
        responsive: true,
        plugins: {
            legend: {
                display: true,
                position: 'top',
            },
            annotation: {
                annotations: [
                    {
                        type: 'line',
                        yMin: 75,
                        yMax: 75,
                        borderColor: 'darkred', // Red and dark color for the threshold line
                        borderWidth: 2,
                        label: {
                            content: '75% Threshold',
                            enabled: true,
                            position: 'start',
                        },
                    },
                ],
            },
        },
        scales: {
            y: {
                ticks: {
                    callback: function(value) {
                        return value + '%'; // Show percentage on the y-axis
                    },
                },
            },
        },
        onClick: (event, elements) => {
            if (elements.length > 0) {
                const index = elements[0].index;
                handleClick(filteredData[index]);
            }
        },
    };

    return (
        <div className="p-6">
            <h1 className="text-2xl font-bold mb-4">Post Office Statistics - {level} Level</h1>
            <div className="flex justify-between mb-4">
                <div>
                    {level === 'State' ? (
                        <p className="text-sm text-gray-600">State Level: Red indicates Poor, Light Green indicates Average, Dark Green indicates Best</p>
                    ) : level === 'Divisional' ? (
                        <p className="text-sm text-gray-600">Divisional Level: Red indicates Poor, Light Green indicates Average, Dark Green indicates Best</p>
                    ) : (
                        <p className="text-sm text-gray-600">City Level: Red indicates Poor, Light Green indicates Average, Dark Green indicates Best</p>
                    )}
                </div>
                <div>
                    <span className="text-sm text-red-500">Red: Poor</span> | 
                    <span className="text-sm text-blue-300"> Light Green: Average </span> | 
                    <span className="text-sm text-green-500">Dark Green: Best</span>
                </div>
            </div>
            <div className="bg-white p-4 rounded shadow">
                {filteredData.length > 0 ? (
                    <Bar data={generateChartData()} options={options} />
                ) : (
                    <p>Loading data...</p>
                )}
            </div>

            {level !== 'State' && (
                <button
                    className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
                    onClick={() => {
                        if (level === 'Divisional') setLevel('State');
                        else if (level === 'City') setLevel('Divisional');
                        setSelected(null);
                    }}
                >
                    Go Back
                </button>
            )}

            {level === 'City' && (
                <button
                    className="mt-4 px-4 py-2 bg-red-500 text-white rounded"
                    onClick={() => alert("Alert sent for this post office city!")}
                >
                    Send Alert for This City
                </button>
            )}
        </div>
    );
}