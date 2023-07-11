# !pip3 install tensorflow
# !pip3 install imgaug
# !pip3 install pixellib
# !pip3 install labelme2coco==0.1.2

#------------------ Train ---------------------------
import pixellib
from pixellib.custom_train import instance_custom_training

vis_img = instance_custom_training()
vis_img.load_dataset("train_DR_m1/")

train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 3, batch_size = 4)
train_maskrcnn.load_pretrained_model("mask_rcnn_coco.h5")
train_maskrcnn.load_dataset("train_DR_m1/")
train_maskrcnn.train_model(num_epochs = 500, augmentation=True,  path_trained_models = "mask_rcnn_models_1/")
print("Finish!")

#------------------ Test ---------------------------

import pixellib
from pixellib.instance import custom_segmentation
import cv2
segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 3, class_names= ["BG", "HE", "EX", "SE"])
segment_image.load_model("mask_rcnn_models_1/mask_rcnn_model500.h5")
segmask, output = segment_image.segmentImage("Test_DR/29bc0e721cfe.png", show_bboxes= True, extract_segmented_objects= False, save_extracted_objects=False, mask_points_values=True)
cv2.imwrite("Test_DR/output_1.png", output)
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
print("=== AREA ====")
print("Number masks: ", len(segmask['masks']))
area = cv2.contourArea(contour)
print("Area: ", area)
# print(output.shape)
print("Finish!")