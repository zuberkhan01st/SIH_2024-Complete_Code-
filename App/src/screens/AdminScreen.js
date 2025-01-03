import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import Toast from 'react-native-toast-message';  // Import the Toast library

export default function AdminScreen({ navigation }) {
  const [reports, setReports] = useState([]);

  // Function to fetch active alerts (reports)
  const fetchReports = async () => {
    try {
      const response = await fetch('https://app-backend-j6s6.onrender.com/api/alerts');
      const data = await response.json();
      if (response.ok) {
        setReports(data);  // Store the fetched reports
        showToast('Reports fetched successfully!');
      } else {
        showToast('No active reports found');
      }
    } catch (error) {
      console.error('Error fetching reports:', error);
      showToast('Failed to fetch reports');
    }
  };

  // Function to show toast
  const showToast = (message) => {
    Toast.show({
      type: 'success',
      position: 'top',
      text1: message,
      visibilityTime: 3000,
      autoHide: true,
    });
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.logoutButton}
        onPress={() => {
          navigation.goBack(); // Go back to the previous screen
          showToast('Logged out successfully');
        }}
      >
        <Text style={styles.logoutButtonText}>Logout</Text>
      </TouchableOpacity>

      <Text style={styles.title}>Post Office Admin Dashboard</Text>

      <View style={styles.buttonsContainer}>
        <TouchableOpacity
          style={styles.button}
          onPress={fetchReports}  // Fetch reports on click
        >
          <Text style={styles.buttonText}>View Reports</Text>
        </TouchableOpacity>
      </View>

      {/* Display reports fetched from the server */}
      <ScrollView style={styles.reportsContainer}>
        {reports.length > 0 ? (
          reports.map((report, index) => (
            <View key={index} style={styles.reportCard}>
              <Text style={styles.reportTitle}>{report.title}</Text>
              <Text style={styles.reportDescription}>{report.description}</Text>
              <Text style={styles.reportDetails}>Location: {report.location}</Text>
              <Text style={styles.reportDetails}>Created by: {report.created_by}</Text>
              <Text style={styles.reportDetails}>Post ID: {report.post_id}</Text>
              <Text style={styles.reportStatus}>Status: {report.status}</Text>
            </View>
          ))
        ) : (
          <Text style={styles.noReportsText}>No active reports found</Text>
        )}
      </ScrollView>

      {/* Toast Component */}
      <Toast />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#e3f2fd',  // Light blue background for a professional theme
    padding: 20,
  },
  logoutButton: {
    backgroundColor: '#1e88e5',  // Post office blue color
    padding: 12,
    borderRadius: 5,
    alignItems: 'center',
    marginBottom: 20,
  },
  logoutButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#0d47a1',  // Dark blue title
  },
  buttonsContainer: {
    width: '100%',
    marginBottom: 20,
    alignItems: 'center',
  },
  button: {
    backgroundColor: '#4caf50',  // Green button for "View Reports"
    padding: 15,
    borderRadius: 10,
    width: '80%',
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  reportsContainer: {
    flex: 1,
    width: '100%',
    marginTop: 10,
  },
  reportCard: {
    backgroundColor: '#fff',
    padding: 15,
    marginBottom: 15,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  reportTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#0d47a1',  // Dark blue title for each report
  },
  reportDescription: {
    fontSize: 16,
    marginBottom: 10,
    color: '#666',
  },
  reportDetails: {
    fontSize: 14,
    color: '#666',
  },
  reportStatus: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#4caf50',
  },
  noReportsText: {
    fontSize: 16,
    textAlign: 'center',
    color: '#888',
  },
});
