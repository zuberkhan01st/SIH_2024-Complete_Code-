import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

export default function ViewInboxScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>View Inbox</Text>
      <TouchableOpacity
        style={styles.button}
        onPress={() => alert('Message Opened')}
      >
        <Text style={styles.buttonText}>Open Message</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.goBack()} // Go back to the PostmasterScreen
      >
        <Text style={styles.buttonText}>Back</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 30,
  },
  button: {
    backgroundColor: '#4caf50',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
