import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Image, ScrollView } from 'react-native';
import { useNavigation } from '@react-navigation/native'; // Import navigation

const HomeScreen = () => {
  const navigation = useNavigation(); // Hook to access the navigation

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.logoContainer}>
        <Image
          source={require('../assets/india_post.png')} // Path to India Post logo
          style={styles.logo}
        />
      </View>

      <Text style={styles.welcomeText}> Welcome to India Post Services</Text>

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('LoginScreen', { userType: 'Postmaster' })} // Pass userType as parameter
        >
          <Text style={styles.buttonText}>Postmaster Login</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('LoginScreen', { userType: 'Admin' })} // Pass userType as parameter
        >
          <Text style={styles.buttonText}>Admin Login</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('LoginScreen', { userType: 'CleaningAssistant' })} // Pass userType as parameter
        >
          <Text style={styles.buttonText}>Cleaning Assistant Login</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.imageContainer}>
        <Image
          source={require('../assets/png-transparent-india-post-office-mail-india-post-payments-bank-logo-government-of-west-bengal-angle-text-logo.png')} // Correct path to the image
          style={styles.image}
        />
      </View>

      <Text style={styles.infoText}>
        @India Post
      </Text>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 1, // Ensures the content is scrollable and grows with the content
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    paddingBottom: 20, // To ensure the content isn't hidden behind the bottom of the screen
  },
  logoContainer: {
    marginTop: 50, // Adjust the margin as needed
    marginBottom: 20,
  },
  logo: {
    width: 200,
    height: 200,
    borderRadius: 15,
    borderWidth: 3,
    borderColor: '#e74c3c',
  },
  welcomeText: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#e74c3c',  // India Post Red
    marginTop: 20,
    marginBottom: 30,
    textAlign: 'center',
  },
  buttonContainer: {
    marginTop: 20,
    width: '80%',
  },
  button: {
    backgroundColor: '#e74c3c',  // India Post Red
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    alignItems: 'center',
    elevation: 5, // Adds shadow for a more raised button effect
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  imageContainer: {
    marginTop: 30,
    alignItems: 'center',
  },
  image: {
    width: 250,
    height: 250,
    borderRadius: 15,
    borderWidth: 3,
    borderColor: '#e74c3c',
  },
  infoText: {
    fontSize: 16,
    color: '#333',
    marginTop: 20,
    textAlign: 'center',
    paddingHorizontal: 20,
    marginBottom: 20,
  },
});

export default HomeScreen;
