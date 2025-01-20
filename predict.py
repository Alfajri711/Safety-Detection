from ultralytics import YOLO
import cv2
import time

model = YOLO("best.pt")

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise Exception("No Camera")

while True:
    ret, image = cam.read()
    if not ret:
        break
    
    _time_mulai = time.time()
    result = model.predict(image, show=True)
    
    print("Waktu", time.time()-_time_mulai)
    #cv2.imshow("Image", image)
    
    _key = cv2.waitKey(1)
    
    # Menyimpan screenshot saat menekan 's'
    if _key == ord('s'):
        screenshot_filename = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(screenshot_filename, image)
        print(f"Screenshot saved as {screenshot_filename}")
    
    # Keluar saat menekan 'q'
    if _key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
