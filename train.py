from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/12/yolo12_LW_BiFPN_LSKA.yaml')  # 从YAML建立一个新模型
    #model = YOLO('./ultralytics/cfg/models/12/yolo12_LW_RepVit.yaml')  # 从YAML建立一个新模型
    results = model.train(
    #data='ultralytics/cfg/datasets/VisDrone.yaml',
    #data='ultralytics/cfg/datasets/VisDrone_DOTA.yaml',
    data='ultralytics/cfg/datasets/VisDrone.yaml',
    device='0',
    epochs=300,
    batch=16, 
    verbose=True,
    patience=0,
    resume=True,
    save=True,
    imgsz=640)
