from ultralytics import YOLO

# -- Initialize YOLO11n model (nano)
model = YOLO('yolo11n.pt')

# -- Train the model
results = model.train(
    data='board_data.yaml',
    lr0=0.01,
    epochs=50,
    batch=32,
    optimizer='SGD',
    momentum=0.9,
    dropout=0.3,
    weight_decay=0.001,
    cos_lr=True,
    device='0',
    imgsz=640,
    box=10.0,
    save_period=1,
    plots=True,
    name='SGD_50_11n'
)