from ultralytics import YOLO
import os
import pandas as pd

def main():
   
    
    # Load the pre-trained model
    filepath = "C:/Users/ROG ZEPHYRUS/MiniProject-Traffic Detection/Predict_Time/" 
    data = pd.read_csv("SignalTiming.csv")
    images = os.listdir(filepath)    
    images = [filename for filename in images if "result" not in filename]
    print(images)
    model = YOLO("C:/Users/ROG ZEPHYRUS/MiniProject-Traffic Detection/runs/detect/train29/weights/best.pt") 
    # Make predictions on new data
    
    for image in images:
        name,extension = os.path.splitext(image)
        file_checker = filepath + name +"result.jpg"
        #print(file_checker)
        if not os.path.exists(file_checker):
            results = model(filepath+image)
            for result in results:
                result_name = name + "result" + extension
                print(result_name)
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification output
                result.save(filepath+result_name)  # save to disk
                names = model.names
                
                HMV = list(names)[list(names.values()).index('HMV')]
                LMV = list(names)[list(names.values()).index('LMV')]
                MW = list(names)[list(names.values()).index('MW')]
                
                HMV_count = results[0].boxes.cls.tolist().count(HMV)
                LMV_count = results[0].boxes.cls.tolist().count(LMV)
                MW_count = results[0].boxes.cls.tolist().count(MW)
                
                row_index = data.index[data['Image'] == float(name)].tolist()
                print(row_index)
                
                data.loc[row_index, 'HMV'] = HMV_count
                data.loc[row_index, 'LMV'] = LMV_count
                data.loc[row_index, 'MW'] = MW_count
                
                updated_csv_file_path = 'SignalTiming.csv'
                data.to_csv(updated_csv_file_path, index=False)

                #print(results[0].boxes.cls.tolist().count(''))
                
                
                
                print("MW:",MW_count)
                print("LMV:",LMV_count)
                print("HMV:",HMV_count)
        else:
            print("Already Done with" + file_checker)
if __name__ == '__main__':
    main()

