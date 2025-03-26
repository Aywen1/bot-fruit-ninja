import time
import cv2
import numpy as np
from ultralytics import YOLO
import dxcam
from pynput.mouse import Controller, Button
from math import hypot

model = YOLO('runs/detect/train3/weights/best.pt')
camera = dxcam.create()
REGION = (0, 0, 1280, 720)

mouse = Controller()

BOMB_SAFE_RADIUS = 150
SLICE_STEPS = 10
SLICE_SLEEP_TIME = 0.0005

active_targets = {}
TARGET_TIMEOUT = 5

def move_and_slice(start_x, start_y, end_x, end_y):
    dx = (end_x - start_x) / SLICE_STEPS
    dy = (end_y - start_y) / SLICE_STEPS

    mouse.position = (int(start_x), int(start_y))
    mouse.press(Button.left)

    for i in range(SLICE_STEPS):
        current_x = start_x + dx * i
        current_y = start_y + dy * i
        mouse.position = (int(current_x), int(current_y))
        time.sleep(SLICE_SLEEP_TIME)

    mouse.release(Button.left)

def move_and_slice_vertical(cx, cy, bomb_positions):
    length = 80
    height = 250

    start_x1 = cx - length // 2
    end_x1 = cx + length // 2
    start_y1 = cy - height // 2
    end_y1 = cy + height // 2

    offset = 20 

    for bx, by in bomb_positions:
        distance = hypot(cx - bx, cy - by)
        if distance < BOMB_SAFE_RADIUS:
            return

    paths = [
        (start_x1, start_y1, end_x1, end_y1),
        (start_x1 + offset, start_y1, end_x1 + offset, end_y1),
        (start_x1 - offset, start_y1, end_x1 - offset, end_y1),
    ]

    for path in paths:
        sx, sy, ex, ey = path
        move_and_slice(sx, sy, ex, ey)

def find_safe_targets(boxes, bomb_positions):
    safe_targets = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if label == "bombe":
            bomb_positions.append((cx, cy))
        else:
            too_close = False
            for bx, by in bomb_positions:
                distance = hypot(cx - bx, cy - by)
                if distance < BOMB_SAFE_RADIUS:
                    too_close = True
                    break
            if not too_close:
                safe_targets.append((cx, cy))

    return safe_targets

print("→ Le bot est lancé, faites [Ctrl+C] pour l'arrêter !")

try:
    while True:
        img = camera.grab(region=REGION)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        results = model.predict(source=img_bgr, conf=0.5, verbose=False)

        bomb_positions = []
        current_targets = []

        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                safe_targets = find_safe_targets(boxes, bomb_positions)
                current_targets.extend(safe_targets)

        updated_targets = {}
        for cx, cy in current_targets:
            key = (cx, cy)
            updated_targets[key] = active_targets.get(key, TARGET_TIMEOUT)

        active_targets = updated_targets

        for (cx, cy), timeout in active_targets.items():
            move_and_slice_vertical(cx, cy, bomb_positions)
            active_targets[(cx, cy)] = timeout - 1

        active_targets = {k: v for k, v in active_targets.items() if v > 0}

        cv2.imshow("Fruit Ninja Bot", img_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Le bot est arrêté.")

cv2.destroyAllWindows()
