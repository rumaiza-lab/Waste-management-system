#define BUFFER_SIZE 32
// Motor driver pins
const int in1 = 3;
const int in2 = 4;
const int in3 = 5;
const int in4 = 6;
// Relay pin (active-low: LOW enables relay)
33
const int relayPin = 8;
// Ultrasonic sensor pins
const int trigPin = 9;
const int echoPin = 10;
const int obstacleThreshold = 20; // Distance threshold in cm
// Function to move forward
void moveForward() {
digitalWrite(in1, HIGH);
digitalWrite(in2, LOW);
digitalWrite(in3, HIGH);
digitalWrite(in4, LOW);
Serial.println("Moving forward");
}
// Function to move backward
void moveBackward() {
digitalWrite(in1, LOW);
digitalWrite(in2, HIGH);
digitalWrite(in3, LOW);
digitalWrite(in4, HIGH);
Serial.println("Moving backward");
}
// Function to turn right
void turnRight() {
digitalWrite(in1, HIGH);
digitalWrite(in2, LOW);
digitalWrite(in3, LOW);
digitalWrite(in4, HIGH);
Serial.println("Turning right");
34
}
// Function to turn left
void turnLeft() {
digitalWrite(in1, LOW);
digitalWrite(in2, HIGH);
digitalWrite(in3, HIGH);
digitalWrite(in4, LOW);
Serial.println("Turning left");
}
// Function to stop movement
void stopMoving() {
digitalWrite(in1, LOW);
digitalWrite(in2, LOW);
digitalWrite(in3, LOW);
digitalWrite(in4, LOW);
Serial.println("Stopping");
}
// Function to measure distance using ultrasonic sensor
int getDistance() {
digitalWrite(trigPin, LOW);
delayMicroseconds(2);
digitalWrite(trigPin, HIGH);
delayMicroseconds(10);
digitalWrite(trigPin, LOW);
long duration = pulseIn(echoPin, HIGH);
int distance = duration * 0.034 / 2; // Convert time to distance in cm
return distance;
}
void setup() {
Serial.begin(9600);
// Motor control pins setup
pinMode(in1, OUTPUT);
pinMode(in2, OUTPUT);
pinMode(in3, OUTPUT);
pinMode(in4, OUTPUT);
// Relay pin setup
pinMode(relayPin, OUTPUT);
// Ultrasonic sensor setup
pinMode(trigPin, OUTPUT);
pinMode(echoPin, INPUT);
// Default movement and disable relay (active-low: HIGH means off)
moveForward();
digitalWrite(relayPin, HIGH);
Serial.println("Arduino Ready");
}
void loop() {
int distance = getDistance();
Serial.print("Distance: ");
Serial.print(distance);
Serial.println(" cm");
// Obstacle detection logic
if (distance > 0 && distance < obstacleThreshold) {
36
Serial.println("Obstacle detected! Changing direction...");
stopMoving();
delay(500); // Small pause before taking action
// Choose a new direction
turnRight();
delay(700); // Adjust turn duration
moveForward();
}
// Processing serial input for waste classification
if (Serial.available() > 0) {
char buffer[BUFFER_SIZE];
int len = Serial.readBytesUntil('\n', buffer, BUFFER_SIZE - 1);
buffer[len] = '\0'; // Null-terminate the string
Serial.print("Received: ");
Serial.println(buffer);
// Process classification result
if (strcmp(buffer, "Biodegradable") == 0) {
digitalWrite(relayPin, LOW); // Activate relay
moveBackward();
} else if (strcmp(buffer, "Non-Biodegradable") == 0) {
digitalWrite(relayPin, HIGH); // Deactivate relay
moveForward();
}
}
}