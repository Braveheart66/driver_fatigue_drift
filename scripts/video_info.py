import cv2, sys
p = sys.argv[1]
cap = cv2.VideoCapture(p)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print('frames:', frames, 'fps:', fps, 'duration_s:', (frames/fps) if fps>0 else 'unknown')
