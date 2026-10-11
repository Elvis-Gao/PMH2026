POLEVISION - QUICK START
========================

EASIEST WAY
  Windows: double-click RUN_ME.bat   (or: python run_polevision.py)
  It asks for the photo, tilt (pitch), and distance, then prints a table of every
  object with confidencePercent, heightMeters, owner, notes, and saves:
    photoName_heights.jpg, photoName_summary.json, allResults.csv (Excel)

FOLDER
  run_polevision.py      easy question-and-answer mode
  RUN_ME.bat             double-click launcher (Windows)
  COLAB_TRAIN_ALL.txt    copy-paste cells to retrain on all 3 datasets (boxes)
  COLAB_TRAIN_SEGMENT.txt  same, but trains an OUTLINE model (best_seg.pt)
  polevision_app.py      the whole system (sensors + AI + heights + report)
  check_model.py         checks how well your trained model detects things
  best.pt                <- PUT YOUR TRAINED MODEL HERE (from Colab)
  photos/                <- put pole photos here
  firmware/pole_sensor_hub/pole_sensor_hub.ino   upload to the Nano

1) INSTALL (once)
     pip install -r requirements.txt

2) CHECK THE MODEL
     python check_model.py photos
   Open photos/checks/ and look at the boxes.

3) HEIGHTS, NO ARDUINO
     python polevision_app.py photos                      every photo, photo-only
     python polevision_app.py photos/pole1.jpg --distance 15 --no-arduino

4) HEIGHTS, WITH ARDUINO
   - Nano runs the sketch with HUMAN_READABLE = false, Serial Monitor CLOSED
     python polevision_app.py --live                      check tilt + distance
     python polevision_app.py photos/pole1.jpg            aim rig at pole, hold still

5) PHONE + ARDUINO (timestamp matching)
   Rig: phone (portrait), Nano and radar taped to ONE board, all facing the pole.
   a) Laptop, Arduino plugged in:   python polevision_app.py --record
   b) Clock check: photograph the laptop screen with the phone (the time column shows),
      copy that photo over, then:   python polevision_app.py --clock-check photos/IMG_xxxx.jpg --shown 02:41:15
      It prints the --clock-offset to use.
   c) Shoot: aim at the pole, wait for STILL on the laptop, take the photo, stay still 2 more seconds.
   d) Copy ORIGINAL photos to photos/ (cable / AirDrop / Google Photos "download") - not chat apps.
   e) python polevision_app.py photos --log sensor_log.csv --clock-offset 2.0
      Each photo gets matched to the reading from the same second (saved as photo.json).

OUTPUT per photo
   pole1_heights.jpg   annotated image
   pole1_report.json   all numbers
   pole1.json          the saved sensor reading (re-runs reuse it)

ACCURACY (the +/- number on every height)
   photo only, assumed level      very rough (several meters)
   + typed distance (--distance)  about +/- 0.5-1 m
   + Arduino tilt and radar       about +/- 0.1-0.2 m
   The hardware is what makes it accurate. Say this in the demo.

OPTIONS
   --imgsz 640     if you trained at 640 instead of 960
   --conf 0.2      show weaker detections
   --model a.pt b.pt   use several models together
