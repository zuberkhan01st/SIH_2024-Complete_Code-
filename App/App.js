import React, { useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import Toast from 'react-native-toast-message';  // Import the Toast library
import HomeScreen from './src/screens/HomeScreen';
import LoginScreen from './src/screens/LoginScreen';
import AdminScreen from './src/screens/AdminScreen';
import PostmasterScreen from './src/screens/PostmasterScreen';
import SweeperScreen from './src/screens/SweeperScreen';
import ManagePostScreen from './src/screens/ManagePostScreen';
import ViewInboxScreen from './src/screens/ViewInboxScreen';

const Stack = createStackNavigator();

export default function App() {
  const [expoPushToken, setExpoPushToken] = useState('');

  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={HomeScreen} options={{ headerShown: false }} />
        <Stack.Screen name="LoginScreen" component={LoginScreen} />
        <Stack.Screen name="AdminScreen" component={AdminScreen} />
        <Stack.Screen name="PostmasterScreen" component={PostmasterScreen} />
        <Stack.Screen name="SweeperScreen" component={SweeperScreen} />
        <Stack.Screen name="ManagePostScreen" component={ManagePostScreen}/>
        <Stack.Screen name="ViewInboxScreen" component={ViewInboxScreen}/>
      </Stack.Navigator>

      {/* Updated Toast Component */}
      <Toast />
    </NavigationContainer>
  );
}
