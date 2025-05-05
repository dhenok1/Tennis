from ultralytics import YOLO

model = YOLO('Models/last.pt')

result = model.predict('Input_Videos\input_video.mp4', save = True)
print(result)
print("Boxes:")
for box in result[0].boxes:
    print(box)





