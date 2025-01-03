import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Dimensions } from 'react-native';
import Toast from 'react-native-toast-message';  // Importing toast message

export default function PostmasterScreen({ navigation }) {
  const [alerts, setAlerts] = useState([]);

  // Fetch alerts from API
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await fetch('https://app-backend-j6s6.onrender.com/api/alerts/');
        const data = await response.json();
        setAlerts(data); // Set the fetched alerts to state
      } catch (error) {
        console.error('Error fetching alerts:', error);
      }
    };

    fetchAlerts();
  }, []); // Run on mount

  // Handle the resolve action for the alert
  const resolveAlert = async (alertId) => {
    try {
      const response = await fetch(`https://app-backend-j6s6.onrender.com/api/alerts/resolve/${alertId}`, {
        method: 'PUT', // Use PUT to resolve the alert
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        // Successfully resolved the alert
        Toast.show({
          type: 'success',
          position: 'top',
          text1: 'Success',
          text2: 'Alert has been resolved',
          visibilityTime: 3000,
          autoHide: true,
          topOffset: 40, // Ensures it doesn't overlap the status bar
        });

        // Optimistically update the UI
        setAlerts((prevAlerts) => 
          prevAlerts.map((alert) =>
            alert._id === alertId ? { ...alert, status: 'Resolved' } : alert
          )
        );
      } else {
        Toast.show({
          type: 'error',
          position: 'top',
          text1: 'Error',
          text2: 'Failed to resolve the alert',
          visibilityTime: 3000,
          autoHide: true,
          topOffset: 40, // Ensures it doesn't overlap the status bar
        });
      }
    } catch (error) {
      console.error('Error resolving alert:', error);
      Toast.show({
        type: 'error',
        position: 'top',
        text1: 'Error',
        text2: 'Failed to resolve the alert',
        visibilityTime: 3000,
        autoHide: true,
        topOffset: 40, // Ensures it doesn't overlap the status bar
      });
    }
  };

  // Handle logout navigation
  const handleLogout = () => {
    navigation.goBack();
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.logoutButton}
        onPress={handleLogout}
      >
        <Text style={styles.logoutButtonText}>Logout</Text> {/* Wrap text within <Text> */}
      </TouchableOpacity>
      
      <Text style={styles.title}>Postmaster Dashboard</Text>
      
      <ScrollView contentContainerStyle={styles.alertsContainer}>
        {alerts.map((alert) => (
          <View key={alert._id} style={styles.alertCard}>
            <Text style={styles.alertTitle}>{alert.title}</Text>
            <Text style={styles.alertContent}>{alert.description}</Text>
            <Text style={styles.alertStatus}>
              Status: {alert.status === 'Resolved' ? 'Resolved' : 'Active'}
            </Text>
            {alert.status !== 'Resolved' && (
              <TouchableOpacity
                style={styles.resolveButton}
                onPress={() => resolveAlert(alert._id)} // Resolve alert on button press
              >
                <Text style={styles.buttonText}>Resolve</Text> {/* Wrap button text within <Text> */}
              </TouchableOpacity>
            )}
          </View>
        ))}
      </ScrollView>
      <Toast ref={(ref) => Toast.setRef(ref)} /> {/* Toast component */}
    </View>
  );
}

const { width } = Dimensions.get('window'); // Get screen width for responsiveness

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'flex-start', // Align content to top to avoid overlap
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    paddingTop: 40, // Give space at the top
  },
  logoutButton: {
    position: 'absolute',
    top: 20,
    right: 20,
    backgroundColor: '#FF5733', // Vibrant red for Indian Post color
    paddingVertical: 12,
    paddingHorizontal: 25,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 6,
  },
  logoutButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#2c3e50',
  },
  alertsContainer: {
    width: width * 0.9, // Make alerts container responsive to screen size
    marginTop: 30,
    paddingBottom: 20,
  },
  alertCard: {
    backgroundColor: '#fff',
    padding: 20,
    marginBottom: 20,
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 5,
  },
  alertTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#2c3e50',
  },
  alertContent: {
    fontSize: 14,
    color: '#555',
    marginBottom: 10,
  },
  alertStatus: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#2980b9',
  },
  resolveButton: {
    backgroundColor: '#2ecc71',
    paddingVertical: 12,
    paddingHorizontal: 25,
    borderRadius: 5,
    alignItems: 'center',
    marginTop: 10,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 60,  // Added marginTop to create space between logout and title
    marginBottom: 20,
    color: '#2c3e50',
  },
  
});
