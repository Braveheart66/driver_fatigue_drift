import time, sys
print('python', sys.version)
# measure cv2 import
start = time.time()
try:
    import cv2
    print('cv2 import time:', time.time() - start)
except Exception as e:
    print('cv2 import failed:', e)
# measure mediapipe import
start = time.time()
try:
    import mediapipe as mp
    print('mediapipe import time:', time.time() - start)
except Exception as e:
    print('mediapipe import failed:', e)
# measure FaceMeshExtractor init
start = time.time()
try:
    from src.extraction.face_mesh import FaceMeshExtractor
    fm = FaceMeshExtractor()
    print('FaceMeshExtractor init time:', time.time() - start, 'enabled:', getattr(fm, 'enabled', False))
except Exception as e:
    print('FaceMeshExtractor init failed:', e)
print('done')
