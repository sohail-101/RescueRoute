// ─────────────────────────────────────────────
//  SMART TRAFFIC LIGHT (4 ROADS)
//  Works with Python sending: '0','1','2','3'
// ─────────────────────────────────────────────

int lights[4][3] = {
  {2, 3, 4},    // NORTH  → Red, Yellow, Green
  {5, 6, 7},    // SOUTH
  {8, 9, 10},   // EAST
  {11, 12, 13}  // WEST
};

int currentLane = -1;  // to avoid repeated updates

// ─────────────────────────────────────────────
void setup() {
  Serial.begin(9600);

  // Set all pins as output
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      pinMode(lights[i][j], OUTPUT);
    }
  }

  setAllRed();
}

// ─────────────────────────────────────────────
void loop() {

  if (Serial.available() > 0) {
    char cmd = Serial.read();

    int lane = getLaneIndex(cmd);

    // Only update if valid AND different
    if (lane != -1 && lane != currentLane) {
      switchLane(lane);
      currentLane = lane;
    }
  }
}

// ─────────────────────────────────────────────
// Convert incoming char → lane index
int getLaneIndex(char c) {
  if (c == '0') return 0; // NORTH
  if (c == '1') return 1; // SOUTH
  if (c == '2') return 2; // EAST
  if (c == '3') return 3; // WEST
  return -1;
}

// ─────────────────────────────────────────────
// Set all lanes RED
void setAllRed() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(lights[i][0], HIGH); // RED ON
    digitalWrite(lights[i][1], LOW);  // YELLOW OFF
    digitalWrite(lights[i][2], LOW);  // GREEN OFF
  }
}

// ─────────────────────────────────────────────
// Switch to a lane (with optional yellow transition)
void switchLane(int lane) {

  // Step 1: turn current green → yellow (optional)
  if (currentLane != -1) {
    digitalWrite(lights[currentLane][2], LOW);  // Green OFF
    digitalWrite(lights[currentLane][1], HIGH); // Yellow ON
    delay(1000); // 1 sec transition
  }

  // Step 2: all red
  setAllRed();
  delay(300);

  // Step 3: selected lane GREEN
  digitalWrite(lights[lane][0], LOW);   // Red OFF
  digitalWrite(lights[lane][2], HIGH);  // Green ON
}

// ─────────────────────────────────────────────