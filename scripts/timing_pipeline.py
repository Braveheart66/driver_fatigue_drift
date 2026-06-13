import time, sys, argparse
import cv2
from src.extraction.aggregate import aggregate_window

p = argparse.ArgumentParser()
p.add_argument('video')
p.add_argument('--max-frames', type=int, default=1800)
args = p.parse_args()

cap = cv2.VideoCapture(args.video)
start = time.time()
count = 0
frame_buffer = []
wins = 0
while count < args.max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    # minimal per-frame feature (fallback)
    features = {'ear':0.0,'mar':0.0,'pitch':0.0,'yaw':0.0,'roll':0.0,
                'au6':0.0,'au12':0.0,'gaze_x':0.0,'gaze_y':0.0}
    frame_buffer.append(features)
    count += 1
    if len(frame_buffer) >= 150:
        _ = aggregate_window(frame_buffer)
        frame_buffer = []
        wins += 1
    if count % 500 == 0:
        elapsed = time.time() - start
        print(f'frames {count} elapsed {elapsed:.2f} fps {count/elapsed:.2f} windows {wins}')

end = time.time()
cap.release()
print('done frames', count, 'elapsed', end-start, 'fps', (count/(end-start)) if end>start else 0.0, 'windows', wins)
