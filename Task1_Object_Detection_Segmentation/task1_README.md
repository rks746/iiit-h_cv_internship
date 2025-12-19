## DETECTION = classification + localization
CNN slides convolutional filters over the image, there are 3 layers - early, middle, deep.  
-Early layers learn edges and corners.  
-Middle layers learn shapes and parts.  
-Deep layers learn object kevel semantics.  
This generates a **feature map** which contains all the information about the image.  
Model predicts a bounding box. Generally, there are overlapping boxes so **Non-Max Suppression (NMS)** is applied to keep the best box and suppress duplicates.  
For each predicted box, the model predicts a **class probability vector** and confidence score. 

## SEGMENTATION - the exact pixels the object belongs to
**Semantic Segmentation** doesn't separate object instances while **Instance Segmentation** (followed by YOLO) separates each object instance.  
CNN extracts feature map and boxes, classes, confidence is found out.  
YOLO uses a prototype mask approach.  
- Network predicts a small set of prototype masks  
- For each detected object, predicts mask coefficients  
- Final mask = weighted sum of prototypes  
Upsampling and thresholding is done to resize mask to original image size.  

## YOLO - You Look Only Once
Entire image is processed in one single pass, hence it is fast. 
