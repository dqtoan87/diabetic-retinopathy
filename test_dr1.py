import pixellib
from pixellib.instance import custom_segmentation
# from pixellib.semantic import semantic_segmentation
# from pixellib.semantic import semantic_segmentation
# from pixellib.instance import instance_segmentation
import cv2
import numpy as np

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 1, class_names= ["BG", "CCCD_BC"])
segment_image.load_model("mask_rcnn_model_500.h5")
# segment_image.segmentImage("test3.jpg", show_bboxes=True, output_image_name="sample_out4.jpg", mask_points_values=True)
segmask, output = segment_image.segmentImage("test1.jpg", show_bboxes= False, extract_segmented_objects= False, save_extracted_objects=False, mask_points_values=True)
cv2.imwrite("output1.jpg", output)
# print("=== MASKS ====")
# print(segmask['masks'])
# print("=== ROIS ====")
# print(segmask['rois'])
# print("=== CLASS ID ====")
# print(segmask['class_ids'])
print("=== SEGMASK ====")
_contour = segmask['masks'][0][0]
print("_contour: ", _contour)
contour = _contour.astype('float32')
print("Goc: ", type(_contour))

print("=== AREA ====")
print("Number masks: ", len(segmask['masks']))
area = cv2.contourArea(contour)
print("Area: ", area)
# print(output.shape)
print("Finish!")
