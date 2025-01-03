import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Animated } from 'react-native';
import Toast from 'react-native-toast-message'; // Import the Toast library

const SweeperScreen = ({ navigation }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false); // Track login status
  const [loading, setLoading] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [associatedPostOffice, setAssociatedPostOffice] = useState('12345'); // Assuming post_id is 12345
  const [newAlert, setNewAlert] = useState(null);
  const [toastYPosition] = useState(new Animated.Value(-100)); // Initial position of toast (offscreen at the top)

  // Fetch all active alerts
  const fetchAlerts = async () => {
    try {
      const response = await fetch(`https://app-backend-j6s6.onrender.com/api/alerts/`);
      const data = await response.json();
      // Sort the alerts by creation date in ascending order (oldest first)
      const sortedAlerts = data.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
      setAlerts(sortedAlerts); // Set sorted alerts in the state

      // Show toast for the last alert if alerts exist
      if (sortedAlerts.length > 0) {
        const lastAlert = sortedAlerts[sortedAlerts.length - 1]; // Most recent alert after sorting
        showToastForNewAlert(lastAlert);
      }
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  };

  const createNewAlert = async () => {
    const alertData = {
      title: 'New Alert for Post Office',
      description: 'Urgent issue reported at post office 12345',
      created_by: 'Sweeper XYZ',
      location: 'Post Office 12345',
      post_id: associatedPostOffice, // Make sure to use the correct post_id
    };

    try {
      const response = await fetch('https://app-backend-j6s6.onrender.com/api/alerts/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(alertData),
      });

      const data = await response.json();
      if (response.status === 201) {
        setAlerts((prevAlerts) => [data, ...prevAlerts]); // Add new alert to the front of the list
        setNewAlert(data); // Set the new alert as the most recent
      } else {
        showToast('Error', 'Failed to add the new alert');
      }
    } catch (error) {
      console.error('Error adding new alert:', error);
      showToast('Error', 'An error occurred while adding the alert');
    }
  };

  // Function to show a toast notification for the most recent alert
  const showToastForNewAlert = (alert) => {
    Animated.timing(toastYPosition, {
      toValue: 50, // Move the toast to the middle of the screen
      duration: 1000, // Duration of the upward animation (1 second)
      useNativeDriver: true,
    }).start(() => {
      // After moving the toast, wait for 2 seconds and then hide it
      setTimeout(() => {
        Animated.timing(toastYPosition, {
          toValue: -80, // Move the toast back offscreen
          duration: 500, // Duration of the downward animation (0.5 second)
          useNativeDriver: true,
        }).start();
      }, 2000); // Wait for 2 seconds before disappearing
    });

    Toast.show({
      type: 'success',
      position: 'top',
      text1: 'New Alert Added!',
      text2: `Title: ${alert.title}, Description: ${alert.description}`,
      visibilityTime: 2000, // Keep visible for 2 seconds after the animation
      autoHide: true, // Auto hide after visibilityTime
      topOffset: 50, // Offset for toast at the top of the screen
    });
  };

  useEffect(() => {
    fetchAlerts(); // Fetch all alerts initially

    const intervalId = setInterval(() => {
      fetchAlerts(); // Fetch alerts every 6 seconds
    }, 8000);

    // Clean up interval on unmount
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array ensures this effect runs once on mount

  useEffect(() => {
    // If there is a new alert, show a toast immediately after adding it
    if (newAlert) {
      showToastForNewAlert(newAlert); // Trigger toast notification for the new alert
    }
  }, [newAlert]); // Only run when a new alert is added

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>Post Office Alerts</Text>
      {!isLoggedIn ? (
        <TouchableOpacity style={styles.loginButton} onPress={() => setIsLoggedIn(true)}>
          <Text style={styles.loginButtonText}>Login</Text>
        </TouchableOpacity>
      ) : (
        <ScrollView style={styles.alertsContainer}>
          {alerts.length > 0 ? (
            alerts.map((alert) => (
              <View key={alert._id} style={styles.alertCard}>
                <Text style={styles.alertTitle}>{alert.title}</Text>
                <Text style={styles.alertDescription}>{alert.description}</Text>
                <Text style={styles.alertInfo}>Created By: {alert.created_by}</Text>
                <Text style={styles.alertInfo}>Location: {alert.location}</Text>
              </View>
            ))
          ) : (
            <Text style={styles.noAlertsText}>No active alerts found.</Text>
          )}
        </ScrollView>
      )}

      {/* Ensure Toast is rendered outside any text structure */}
      <Toast />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fafafa', // Light background color
    padding: 20,
    alignItems: 'center',
  },
  heading: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#e74c3c',
    marginVertical: 20,
    textAlign: 'center',
  },
  loginButton: {
    backgroundColor: '#e74c3c',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginTop: 20,
  },
  loginButtonText: {
    color: '#fff',
    fontSize: 18,
  },
  alertsContainer: {
    width: '100%',
    marginTop: 20,
  },
  alertCard: {
    width: '100%',
    backgroundColor: '#fff',
    padding: 15,
    marginBottom: 15,
    borderRadius: 12,
    elevation: 4, // Adds shadow for a modern card effect
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
  },
  alertTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  alertDescription: {
    fontSize: 16,
    color: '#34495e',
    marginVertical: 5,
  },
  alertInfo: {
    fontSize: 14,
    color: '#7f8c8d',
    marginVertical: 2,
  },
  noAlertsText: {
    textAlign: 'center',
    fontSize: 18,
    color: '#7f8c8d',
  },
});

export default SweeperScreen;
