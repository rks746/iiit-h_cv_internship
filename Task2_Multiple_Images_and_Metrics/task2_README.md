YOLO does not scan object-by-object. It predicts hundreds to thousands of candidates at once.

The model will produce separate predictions for each by parallel prediction:  
- Each object → its own bounding box
- Its own class label
- Its own confidence score
- Its own segmentation mask (if seg model)  

When two objects are overlapping, the model may not detect or segment both the objects.  
NMS keeps the higher confidence box and deletes the lower confidence boxes.  
There are ways to fix it like soft-NMS, DIoU, class-wise NMS, anchor-free detectors, ditch boxes. 
But YOLO still uses NMS because it is fast, simple and GPU-friendly.  

## Evaluation Metrics 
Accuracy - proportion of all predictions that are correct  
Precision - among everything predicted positive, how many were actually positive.  
Recall - true positive rate = among all actual positives, how many were found  
F1 score - single number that balances precision and recall (harmonic mean)  
Confusion matrix - a foundational table showing TP, TN, FP, FN counts, from which precision/recall are derived  
Confidence threshold sets the probability cutoff for predictions (e.g., >0.5 for positive), tuning precision-recall trade-off.  
Confidence distribution examines prediction probabilities' calibration (e.g., via reliability diagrams), revealing if high-confidence predictions are trustworthy.  
Macro precision/recall/F1 with `average='macro'` compute per-class metrics first (precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2_prec_rec/(prec+rec) for each class), then take the unweighted arithmetic mean across all classes—treats rare classes equally.  