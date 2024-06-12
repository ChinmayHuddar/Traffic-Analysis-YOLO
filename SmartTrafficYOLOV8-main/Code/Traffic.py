from ultralytics import YOLO

# Load a model
def Traffic():
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    model.train(data="data.yaml", epochs=100, batch=2)

if __name__ == '__main__':
    Traffic()        