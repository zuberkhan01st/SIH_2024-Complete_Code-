#include <SoftwareSerial.h>

// Timer interval in milliseconds
const unsigned long interval = 5000; // 5 seconds
unsigned long previousMillis = 0;

// Define the Post Office ID
String postOfficeID = "PO123";

void setup() {
  // Start serial communication
  Serial.begin(9600);

  // Initialize the built-in LED
  pinMode(LED_BUILTIN, OUTPUT);

  // Initially turn on the LED to indicate that the system is ready
  digitalWrite(LED_BUILTIN, HIGH);

  // Initial message
  Serial.println("Arduino ready to send signals...");
}

void loop() {
  // Current time
  unsigned long currentMillis = millis();

  // Check if the interval has passed (every 5 seconds)
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    // Send the request to Python with the Post Office ID
    String message = "REQUEST_IMAGE_UPLOAD:" + postOfficeID;
    Serial.println(message);

    // Keep the LED on while the request is sent
    digitalWrite(LED_BUILTIN, HIGH);
  }

  // Check for response from Python
  if (Serial.available() > 0) {
    String response = Serial.readStringUntil('\n');
    response.trim(); // Remove any trailing whitespace

    if (response == "UPLOAD_SUCCESS") {
      // Blink the LED to indicate success
      blinkLED();
    }
  }
}

// Function to blink the LED for success indication
void blinkLED() {
  // Turn LED on for 500ms (indicating success)
  digitalWrite(LED_BUILTIN, HIGH);
  delay(500); // LED on for 500ms
  digitalWrite(LED_BUILTIN, LOW);
  delay(500); // LED off for 500ms (blink)
}
