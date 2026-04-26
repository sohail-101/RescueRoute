/*
 * traffic_lights.ino
 * ═══════════════════════════════════════════════════════════════════
 * 4-Way Intersection Traffic Light Controller
 * Arduino Uno — receives commands from Python via USB Serial
 *
 * SERIAL PROTOCOL  (9600 baud, newline-terminated)
 * ─────────────────────────────────────────────────────────────────
 * Single-character commands:
 *   '0'  →  NORTH gets GREEN  (others RED)
 *   '1'  →  SOUTH gets GREEN  (others RED)
 *   '2'  →  EAST  gets GREEN  (others RED)
 *   '3'  →  WEST  gets GREEN  (others RED)
 *   'X'  →  All RED  (safe / default state)
 *   'P'  →  Ping — Arduino replies "OK\n" for connection test
 *
 * Full-string commands (matches Python communication.py protocol):
 *   "SIGNAL:NORTH:GREEN\n"
 *   "SIGNAL:SOUTH:RED\n"
 *   "SIGNAL:EAST:YELLOW\n"
 *   "SIGNAL:WEST:GREEN\n"
 *
 * Acknowledgement:
 *   After every successful command the Arduino sends back:
 *   "ACK:<LANE>:<STATE>\n"   e.g. "ACK:NORTH:GREEN\n"
 *   On ping: "OK\n"
 *   On invalid input: "ERR:<raw>\n"
 *
 * PIN MAPPING
 * ─────────────────────────────────────────────────────────────────
 *   NORTH  Red→2   Yellow→3   Green→4
 *   SOUTH  Red→5   Yellow→6   Green→7
 *   EAST   Red→8   Yellow→9   Green→10
 *   WEST   Red→11  Yellow→12  Green→13
 *
 * All LED cathodes connected to GND via 220 Ω resistors.
 *
 * YELLOW TRANSITION
 * ─────────────────────────────────────────────────────────────────
 *   When a lane switch is requested, the currently GREEN lane
 *   briefly shows YELLOW for YELLOW_MS milliseconds before going RED
 *   and the new lane turns GREEN.
 *   Set YELLOW_MS = 0 to disable the transition.
 *
 * STARTUP DEFAULT
 * ─────────────────────────────────────────────────────────────────
 *   All lanes RED on power-up.
 *   NORTH goes GREEN after STARTUP_DELAY_MS to signal readiness.
 * ═══════════════════════════════════════════════════════════════════
 */

// ──────────────────────────────────────────────────────────────────
//  Configuration
// ──────────────────────────────────────────────────────────────────
#define NUM_LANES       4
#define YELLOW_MS       2500   // ms for yellow transition; 0 to disable
#define STARTUP_DELAY_MS 1500  // ms after boot before NORTH goes green
#define BAUD_RATE       9600
#define CMD_BUF_SIZE    64     // max serial command length

// Lane indices
#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

// Signal indices within each lane
#define SIG_RED    0
#define SIG_YELLOW 1
#define SIG_GREEN  2

// ──────────────────────────────────────────────────────────────────
//  Pin table  [lane][signal]
// ──────────────────────────────────────────────────────────────────
const uint8_t PIN[NUM_LANES][3] = {
  //  RED   YELLOW  GREEN
  {   2,    3,      4   },   // NORTH
  {   5,    6,      7   },   // SOUTH
  {   8,    9,      10  },   // EAST
  {  11,   12,      13  },   // WEST
};

// Human-readable lane names (stored in flash to save RAM)
const char* const LANE_NAME[NUM_LANES] = {
  "NORTH", "SOUTH", "EAST", "WEST"
};

// State names
const char* const STATE_NAME[3] = {
  "RED", "YELLOW", "GREEN"
};

// ──────────────────────────────────────────────────────────────────
//  Runtime state
// ──────────────────────────────────────────────────────────────────
int8_t  g_activeGreen = -1;   // index of the currently GREEN lane (-1 = none)
char    g_cmdBuf[CMD_BUF_SIZE];
uint8_t g_cmdLen = 0;

// ──────────────────────────────────────────────────────────────────
//  Low-level LED helpers
// ──────────────────────────────────────────────────────────────────

/*
 * setLane() — set one lane to RED / YELLOW / GREEN.
 * sigIdx: SIG_RED=0, SIG_YELLOW=1, SIG_GREEN=2
 * All three pins are written so exactly one is HIGH.
 */
void setLane(uint8_t lane, uint8_t sigIdx) {
  for (uint8_t s = 0; s < 3; s++) {
    digitalWrite(PIN[lane][s], (s == sigIdx) ? HIGH : LOW);
  }
}

// Set every lane to RED.
void allRed() {
  for (uint8_t i = 0; i < NUM_LANES; i++) {
    setLane(i, SIG_RED);
  }
}

// ──────────────────────────────────────────────────────────────────
//  Lane activation with optional Yellow transition
// ──────────────────────────────────────────────────────────────────

/*
 * activateLane(newGreen)
 *
 * Sequence:
 *   1. If a different lane is currently GREEN → set it to YELLOW
 *   2. Wait YELLOW_MS
 *   3. Set that lane to RED
 *   4. Set newGreen lane to GREEN
 *   5. Keep all other lanes RED throughout
 *
 * If newGreen == g_activeGreen, this is a no-op (already active).
 */
void activateLane(uint8_t newGreen) {
  if ((int8_t)newGreen == g_activeGreen) return;   // already active

  // Step 1 — flash yellow on the outgoing green lane
  if (g_activeGreen >= 0 && YELLOW_MS > 0) {
    setLane((uint8_t)g_activeGreen, SIG_YELLOW);
    delay(YELLOW_MS);
    setLane((uint8_t)g_activeGreen, SIG_RED);
  }

  // Step 2 — force all lanes RED, then light the chosen green
  allRed();
  setLane(newGreen, SIG_GREEN);
  g_activeGreen = (int8_t)newGreen;

  // Acknowledge over serial
  Serial.print("ACK:");
  Serial.print(LANE_NAME[newGreen]);
  Serial.print(":GREEN\n");
}

// ──────────────────────────────────────────────────────────────────
//  Command parsers
// ──────────────────────────────────────────────────────────────────

/*
 * handleSingleChar()
 * Interprets single-byte commands:
 *   '0'–'3' → activate lane 0–3
 *   'X','x' → all RED
 *   'P','p' → ping reply
 */
void handleSingleChar(char c) {
  if (c >= '0' && c <= '3') {
    activateLane((uint8_t)(c - '0'));
    return;
  }
  if (c == 'X' || c == 'x') {
    allRed();
    g_activeGreen = -1;
    Serial.print("ACK:ALL:RED\n");
    return;
  }
  if (c == 'P' || c == 'p') {
    Serial.print("OK\n");
    return;
  }
}

/*
 * handleFullCommand()
 * Parses "SIGNAL:<LANE>:<STATE>" commands coming from Python.
 *
 * Accepted LANEs : NORTH SOUTH EAST WEST
 * Accepted STATEs: RED YELLOW GREEN
 *
 * Effect:
 *   GREEN  → activateLane (handles yellow transition + all-red logic)
 *   RED    → only sets that specific lane RED (no transition needed)
 *   YELLOW → only sets that specific lane YELLOW
 *
 * The Python side sends individual per-lane commands, so we honour
 * each one independently.  The GREEN command is the one that triggers
 * the full switching sequence.
 */
void handleFullCommand(const char* cmd) {
  // Expect "SIGNAL:XXXX:YYYYY"
  if (strncmp(cmd, "SIGNAL:", 7) != 0) {
    Serial.print("ERR:"); Serial.print(cmd); Serial.print("\n");
    return;
  }

  // Parse lane name (after "SIGNAL:")
  const char* laneStart = cmd + 7;
  const char* colon2    = strchr(laneStart, ':');
  if (colon2 == nullptr) {
    Serial.print("ERR:"); Serial.print(cmd); Serial.print("\n");
    return;
  }

  // Extract lane string
  char laneBuf[8] = {0};
  uint8_t laneLen = (uint8_t)(colon2 - laneStart);
  if (laneLen == 0 || laneLen >= sizeof(laneBuf)) {
    Serial.print("ERR:LANE_LEN\n"); return;
  }
  strncpy(laneBuf, laneStart, laneLen);

  // Match lane name
  int8_t laneIdx = -1;
  for (uint8_t i = 0; i < NUM_LANES; i++) {
    if (strcmp(laneBuf, LANE_NAME[i]) == 0) {
      laneIdx = (int8_t)i; break;
    }
  }
  if (laneIdx < 0) {
    Serial.print("ERR:LANE:"); Serial.print(laneBuf); Serial.print("\n");
    return;
  }

  // Extract state string (after second colon)
  const char* stateStr = colon2 + 1;

  if (strcmp(stateStr, "GREEN") == 0) {
    activateLane((uint8_t)laneIdx);
  }
  else if (strcmp(stateStr, "RED") == 0) {
    // Only update this lane if it isn't the currently active green
    // (Python sometimes sends RED for lanes that are already red — safe to ignore)
    if (g_activeGreen == laneIdx) {
      setLane((uint8_t)laneIdx, SIG_RED);
      g_activeGreen = -1;
    }
    // Else: already red, nothing to do
    Serial.print("ACK:"); Serial.print(laneBuf);
    Serial.print(":RED\n");
  }
  else if (strcmp(stateStr, "YELLOW") == 0) {
    setLane((uint8_t)laneIdx, SIG_YELLOW);
    Serial.print("ACK:"); Serial.print(laneBuf);
    Serial.print(":YELLOW\n");
  }
  else {
    Serial.print("ERR:STATE:"); Serial.print(stateStr); Serial.print("\n");
  }
}

/*
 * processCommand()
 * Route accumulated command string to the right handler.
 * Strips trailing \r\n before processing.
 */
void processCommand(char* buf, uint8_t len) {
  // Strip CR / LF
  while (len > 0 && (buf[len-1] == '\r' || buf[len-1] == '\n')) {
    buf[--len] = '\0';
  }
  if (len == 0) return;

  if (len == 1) {
    handleSingleChar(buf[0]);
  } else {
    handleFullCommand(buf);
  }
}

// ──────────────────────────────────────────────────────────────────
//  setup()
// ──────────────────────────────────────────────────────────────────
void setup() {
  // Initialise all LED pins as outputs
  for (uint8_t lane = 0; lane < NUM_LANES; lane++) {
    for (uint8_t sig = 0; sig < 3; sig++) {
      pinMode(PIN[lane][sig], OUTPUT);
    }
  }

  // Safe default: everything RED
  allRed();

  // Open serial port
  Serial.begin(BAUD_RATE);

  // Brief startup delay then signal readiness
  delay(STARTUP_DELAY_MS);

  // NORTH green by default — shows the system is alive
  activateLane(NORTH);

  Serial.print("READY\n");
}

// ──────────────────────────────────────────────────────────────────
//  loop()
// ──────────────────────────────────────────────────────────────────
void loop() {
  // Non-blocking serial read — accumulate into g_cmdBuf until '\n'
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    // Newline signals end of command
    if (c == '\n') {
      g_cmdBuf[g_cmdLen] = '\0';
      processCommand(g_cmdBuf, g_cmdLen);
      g_cmdLen = 0;   // reset buffer
      continue;
    }

    // Single-byte commands can be processed immediately (no newline needed)
    // This allows Python to send bare '0'/'1'/'2'/'3' without '\n'
    if (g_cmdLen == 0 && (c >= '0' && c <= '3')
                      || (c == 'X') || (c == 'x')
                      || (c == 'P') || (c == 'p')) {
      // Peek ahead — if no more bytes arrive in 5 ms, treat as single-char cmd
      delay(5);
      if (Serial.available() == 0 || Serial.peek() == '\n' || Serial.peek() == '\r') {
        if (Serial.available() > 0 &&
            (Serial.peek() == '\n' || Serial.peek() == '\r')) {
          Serial.read();   // consume the trailing newline
        }
        handleSingleChar(c);
        g_cmdLen = 0;
        continue;
      }
      // If more bytes follow, fall through to buffered accumulation
    }

    // Accumulate character (guard against buffer overflow)
    if (g_cmdLen < CMD_BUF_SIZE - 1) {
      g_cmdBuf[g_cmdLen++] = c;
    } else {
      // Buffer overflow — discard and reset
      Serial.print("ERR:OVERFLOW\n");
      g_cmdLen = 0;
    }
  }
}
