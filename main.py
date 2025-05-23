import cv2
import face_recognition
import numpy as np
from playsound import playsound

video = cv2.VideoCapture(0)

while True:
  ret, frame = video.read()

  fairy_overlay = cv2.imread('assets/fairy.png')
  fairy_overlay = cv2.resize(fairy_overlay, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
  h, w = fairy_overlay.shape[:2]

  shapes = np.zeros_like(frame, np.uint8)
  shapes[frame.shape[0]-h:, frame.shape[1]-w:] = fairy_overlay
  mask = shapes.astype(bool)

  frame_video = frame.copy()
  frame_video[mask] = cv2.addWeighted(frame_video, 1, shapes, 0.3, 0)[mask]

  playsound('assets/fairy.mp3', False)

  cv2.imshow('Dad prank', frame_video)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video.release()
cv2.destroyAllWindows()