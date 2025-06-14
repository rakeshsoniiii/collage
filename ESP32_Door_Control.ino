#include <Adafruit_Fingerprint.h>
#include <HardwareSerial.h>

#define RELAY_1 19     // Relay pin
#define BUZZER_PIN 14  // Buzzer pin

HardwareSerial mySerial(2);
Adafruit_Fingerprint finger = Adafruit_Fingerprint(&mySerial);

bool doorLocked = true;
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 3000; // 3 seconds timeout

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_1, OUTPUT);
  digitalWrite(RELAY_1, HIGH);
  pinMode(BUZZER_PIN, OUTPUT);
  
  mySerial.begin(57600, SERIAL_8N1, 16, 17);
  finger.begin(57600);
  
  if (!finger.verifyPassword()) {
    Serial.println("Sensor not found!");
    while(1);
  }
}

void loop() {
  // Check for commands from Python
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "UNLOCK") {
      unlockDoor();
    } else if (command == "LOCK") {
      lockDoor();
    }
  }
  
  // Check fingerprint sensor
  getFingerprintID();
  
  // Auto-lock after timeout
  if (!doorLocked && (millis() - lastCommandTime > COMMAND_TIMEOUT)) {
    lockDoor();
  }
  
  delay(50);
}

uint8_t getFingerprintID() {
  uint8_t p = finger.getImage();
  if (p != FINGERPRINT_OK) return p;
  
  p = finger.image2Tz();
  if (p != FINGERPRINT_OK) return p;

  p = finger.fingerFastSearch();
  if (p != FINGERPRINT_OK) {
    Serial.println("Unknown fingerprint!");
    playUnknownSound();  // 2-second alert sound
    return p;
  }

  Serial.print("Access granted! ID:");
  Serial.println(finger.fingerID);
  playSuccessSound();   // 1-second happy sound
  unlockDoor();
  return finger.fingerID;
}

void unlockDoor() {
  if (doorLocked) {
    digitalWrite(RELAY_1, LOW);  // Activate relay
    playSuccessSound();
    doorLocked = false;
    lastCommandTime = millis();
    Serial.println("Door unlocked");
  }
}

void lockDoor() {
  if (!doorLocked) {
    digitalWrite(RELAY_1, HIGH);  // Deactivate relay
    playLockSound();
    doorLocked = true;
    Serial.println("Door locked");
  }
}

void playSuccessSound() {
  // Short, pleasant ascending triple beep (1 second total)
  unsigned long startTime = millis();
  while(millis() - startTime < 1000) {  // Run for 1 second
    tone(BUZZER_PIN, 523, 100);  // C5
    delay(120);
    tone(BUZZER_PIN, 659, 100);  // E5
    delay(120);
    tone(BUZZER_PIN, 784, 100);  // G5
    delay(120);
    noTone(BUZZER_PIN);
    delay(300);
  }
}

void playUnknownSound() {
  // Harsh, alternating low-high beeps (2 seconds total)
  unsigned long startTime = millis();
  while(millis() - startTime < 2000) {  // Run for 2 seconds
    tone(BUZZER_PIN, 200, 100);  // Low warning tone
    delay(150);
    tone(BUZZER_PIN, 800, 100);  // High alert tone
    delay(150);
  }
  noTone(BUZZER_PIN);
}

void playLockSound() {
  // Double beep for locking
  tone(BUZZER_PIN, 400, 100);
  delay(200);
  tone(BUZZER_PIN, 400, 100);
  delay(200);
  noTone(BUZZER_PIN);
}