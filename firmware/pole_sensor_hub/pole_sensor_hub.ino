// PoleVision sensor hub - Arduino Nano 33 BLE / Nano 33 BLE Sense Rev2
// Board: "Arduino Nano 33 BLE" (Arduino Mbed OS Nano Boards). Library: Arduino_BMI270_BMM150
// Wiring (DFRobot SEN0306 radar): pin 1 -> 5V supply +, pin 2 -> supply - AND Nano GND,
//                                 pin 4 (TX) -> Nano RX0 (D0), pins 3/5/6 unconnected (single target)
// Output: one JSON line per second with averaged tilt, a STILL flag, and median radar distance.

#include <Arduino_BMI270_BMM150.h>

// ================= SETTINGS =================
const bool  HUMAN_READABLE = false;   // keep false for polevision_app.py
const unsigned long PERIOD_MS = 1000; // one reading per second
const int   FWD_AXIS = 1;             // IMU axis out of the camera lens: 0=x 1=y 2=z
const int   UP_AXIS  = 2;             // IMU axis toward the top of the image: 0=x 1=y 2=z
const bool  FLIP_PITCH = false;       // true if tilting UP shows negative pitch
const float PITCH_OFFSET = 0.0;       // pitch shown when the rig is truly level
const float ROLL_OFFSET  = 0.0;
const float STILL_GYRO_DPS = 3.0;
const float STILL_PITCH_RANGE = 0.5;
// ============================================

uint8_t frame[8];
int fidx = 0;
unsigned long radarLastMs = 0;
bool radarSeen = false;
float distBuf[32];
int distN = 0;

void feedRadar(uint8_t b) {
  if (fidx < 3) { if (b == 0xFF) frame[fidx++] = b; else fidx = 0; return; }
  if (fidx == 3 && b == 0xFF) return;
  frame[fidx++] = b;
  if (fidx == 8) {
    if (frame[5] == 0 && frame[6] == 0 && frame[7] == 0) {
      int cm = (frame[3] << 8) | frame[4];
      if (cm > 0 && distN < 32) distBuf[distN++] = cm / 100.0f;
      radarLastMs = millis();
      radarSeen = true;
    }
    fidx = 0;
  }
}

float median(float *v, int n) {
  for (int i = 1; i < n; i++) { float k = v[i]; int j = i - 1; while (j >= 0 && v[j] > k) { v[j + 1] = v[j]; j--; } v[j + 1] = k; }
  return (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

float sumA[3] = {0, 0, 0};
int nAcc = 0;
float gMax = 0, pMin = 999, pMax = -999;
unsigned long lastOut = 0, seq = 0;

float pitchOf(float a[3]) {
  float g = sqrtf(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
  if (g < 0.01f) g = 1.0f;
  float p = degrees(asinf(constrain(a[FWD_AXIS] / g, -1.0f, 1.0f)));
  return (FLIP_PITCH ? -p : p) - PITCH_OFFSET;
}

float rollOf(float a[3]) {
  int side = 3 - FWD_AXIS - UP_AXIS;
  return degrees(atan2f(a[side], a[UP_AXIS])) - ROLL_OFFSET;
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(57600);
  if (!IMU.begin()) {
    while (true) { Serial.println("{\"err\":\"imu_init_failed\"}"); delay(1000); }
  }
}

void loop() {
  while (Serial1.available()) feedRadar((uint8_t)Serial1.read());

  float x, y, z;
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);
    sumA[0] += x; sumA[1] += y; sumA[2] += z; nAcc++;
    float one[3] = {x, y, z};
    float p = pitchOf(one);
    if (p < pMin) pMin = p;
    if (p > pMax) pMax = p;
  }
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(x, y, z);
    float gy = sqrtf(x*x + y*y + z*z);
    if (gy > gMax) gMax = gy;
  }

  unsigned long now = millis();
  if (now - lastOut >= PERIOD_MS && nAcc > 0) {
    lastOut = now;
    seq++;
    float a[3] = { sumA[0] / nAcc, sumA[1] / nAcc, sumA[2] / nAcc };
    float pitch = pitchOf(a), roll = rollOf(a);
    float pRange = pMax - pMin;
    bool still = (gMax < STILL_GYRO_DPS) && (pRange < STILL_PITCH_RANGE);
    int nd = distN;
    float dist = (nd > 0) ? median(distBuf, nd) : -1.0f;

    if (HUMAN_READABLE) {
      Serial.print("#"); Serial.print(seq); Serial.print("  pitch "); Serial.print(pitch, 1);
      Serial.print("  roll "); Serial.print(roll, 1); Serial.print(still ? "  STILL" : "  moving");
      Serial.print("  |  distance ");
      if (nd > 0) { Serial.print(dist, 2); Serial.println(" m"); } else Serial.println("none");
    } else {
      Serial.print("{\"seq\":"); Serial.print(seq);
      Serial.print(",\"t_ms\":"); Serial.print(now);
      Serial.print(",\"pitch\":"); Serial.print(pitch, 2);
      Serial.print(",\"roll\":"); Serial.print(roll, 2);
      Serial.print(",\"still\":"); Serial.print(still ? "true" : "false");
      Serial.print(",\"pitch_range\":"); Serial.print(pRange, 2);
      Serial.print(",\"dist_m\":"); Serial.print(dist, 2);
      Serial.print(",\"dist_n\":"); Serial.print(nd);
      Serial.println("}");
    }
    sumA[0] = sumA[1] = sumA[2] = 0; nAcc = 0;
    gMax = 0; pMin = 999; pMax = -999; distN = 0;
  }
}
