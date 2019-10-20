import os
import json
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from keras.models import load_model
from .utils.utils import get_yolo_boxes, makedirs, compute_overlap
from .utils.bbox import draw_boxes, draw_boxes1

class Predict():
    def __init__(self, config_path, input_path, output_path):
        self.config_path = config_path
        self.input_path = input_path
        self.output_path = output_path
        #print(self.config_path)
        with open("training\\"+self.config_path) as config_buffer:
            self.config = json.load(config_buffer)
        self.infer_model = load_model("training\\"+self.config['train']['saved_weights_name'])

    def run(self):

        makedirs(self.output_path)

        ###############################
        #   Set some parameter
        ###############################
        net_h, net_w = 256, 256 # a multiple of 32, the smaller the faster
        obj_thresh, nms_thresh = 0.85, 0.25

        ###############################
        #   Load the model
        ###############################
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['train']['gpus']

        ###############################
        #   Predict bounding boxes
        ###############################
        if 'webcam' in self.input_path: # do detection on the first webcam
            video_reader = cv2.VideoCapture(0)

            # the main loop
            batch_size  = 1
            images      = []
            while True:
                ret_val, image = video_reader.read()
                if ret_val == True: images += [image]

                if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                    batch_boxes = get_yolo_boxes(self.infer_model, images, net_h, net_w, self.config['model'][
                        'anchors'], obj_thresh, nms_thresh)

                    for i in range(len(images)):
                        draw_boxes(images[i], batch_boxes[i], '1', obj_thresh)
                        cv2.imshow('video with bboxes', images[i])
                    images = []
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
            cv2.destroyAllWindows()
        elif self.input_path[-4:] == '.mp4': # do detection on a video
            video_out = self.output_path + self.input_path.split('/')[-1]
            video_reader = cv2.VideoCapture(self.input_path)

            nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

            video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'MPEG'),
                                   50.0,
                                   (frame_w, frame_h))
            # the main loop
            batch_size  = 1
            images      = []
            start_point = 0 #%
            show_window = False
            for i in tqdm(range(nb_frames)):
                _, image = video_reader.read()

                if (float(i+1)/nb_frames) > start_point/100.:
                    images += [image]

                    if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                        # predict the bounding boxes
                        batch_boxes = get_yolo_boxes(self.infer_model, images, net_h, net_w, self.config['model'][
                            'anchors'], obj_thresh, nms_thresh)

                        for i in range(len(images)):
                            # draw bounding boxes on the image using labels
                            draw_boxes(images[i], batch_boxes[i], '1', obj_thresh)

                            # show the video with detection bounding boxes
                            if show_window: cv2.imshow('video with bboxes', images[i])

                            # write result to the output video
                            video_writer.write(images[i])
                        images = []
                    if show_window and cv2.waitKey(1) == 27: break  # esc to quit

            if show_window: cv2.destroyAllWindows()
            video_reader.release()
            video_writer.release()
        else: # do detection on an image
            # the main loop
            # Predicting Inference image i.e. semi occupied slots with the bounding boxes
            predicted_image_path = 'data\\test.jpg'
            predicted_image = cv2.imread(predicted_image_path)
            predicted_parking_boxes = get_yolo_boxes(self.infer_model, [predicted_image], net_h, net_w,
                                                     self.config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # Predicting Fully occupied car parking slot with the bounding boxes
            full_parking_image = cv2.imread('data\\fullparking.jpg')
            full_parking_boxes = get_yolo_boxes(self.infer_model, [full_parking_image], net_h, net_w,
                                                self.config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # converting into ndArray for Fully car parked and Inference image
            pred_boxes = np.array(
                [[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in predicted_parking_boxes])
            fullpark_boxes = np.array(
                [[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in full_parking_boxes])

            #using K-Means indentified minimum IOU
            iou_threshold = 0.5

            pred_occupied_boxes = self.create_boxes(pred_boxes, fullpark_boxes, 1, iou_threshold)
            pred_empty_boxes = self.create_boxes(fullpark_boxes, pred_boxes, 0, iou_threshold)

            i = 0
            for d in pred_occupied_boxes:
                self.crop(predicted_image_path, (d[0], d[1], d[2], d[3]), 'data\\output\\predicted\\{0}.png'.format(i))
                i = i + 1

            predicted_image1 = cv2.imread(predicted_image_path)

            # draw bounding boxes on the image using labels
            predicted_image1 = draw_boxes1(predicted_image1.copy(), pred_empty_boxes, 1)

            # draw bounding boxes on the image using labels
            predicted_image1 = draw_boxes1(predicted_image1.copy(), pred_occupied_boxes, 0)

            # write the image with bounding boxes to file
            cv2.imwrite('data\\output\\output_predicted_' + predicted_image_path.split('\\')[-1],
                        (predicted_image1).astype('uint8'))

            # output Result setting
            modified_pred_occupied_boxes = [[d[5], d[6], d[7], d[8], d[9]] for d in pred_occupied_boxes]
            modified_pred_empty_boxes = [[d[5], d[6], d[7], d[8], d[9]] for d in pred_empty_boxes]

            # Predicted Data file
            modified_pred_occupied_boxes.extend(modified_pred_empty_boxes)
            boxes = modified_pred_occupied_boxes

            # Create dict for Empty and Occupied slots
            dic = {}
            dic['filename'] = predicted_image_path
            dic['parking_lots'] = boxes

            # print(dic)

            # Dump data dict to json
            self.writeToJSONFile('data\\annotations\\', 'annotations', dic)


    def create_boxes(self, source, destination, occupancy_flag, iou_threshold):
        filter_overlapping_box = []
        detections = source
        annotations = destination
        detected_annotations = []
        # print(" Total Detection - {0}".format(len(detections)))
        for d in detections:
            overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]
            # print("overlaps {0}, assigned_annotation {1}, max_overlap {2}, iou_threshold {3}".format(overlaps,assigned_annotation,max_overlap,iou_threshold))
            if (occupancy_flag == 1):
                # occupied parking slot
                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    # print("predicted correctly {0}".format(_count))
                    item_present = False
                    detected_annotations.append(assigned_annotation)
                    for box in filter_overlapping_box:
                        if (d[0] == box[0] and d[1] == box[1] and d[2] == box[2] and d[3] == box[3] and d[4] == box[4]):
                            item_present = True
                            break

                    if not item_present:
                        pose = 2
                        height = int(d[3]) - int(d[1])
                        width = int(d[2]) - int(d[0])
                        if (height > width):
                            pose = 0
                        filter_overlapping_box.append(
                            [int(d[0]), int(d[1]), int(d[2]), int(d[3]), d[4], int((d[0] + d[2]) / 2),
                             int((d[1] + d[3]) / 2), occupancy_flag, 0, pose])
            else:
                # Empty parking slot
                if max_overlap < iou_threshold:
                    # print("predicted correctly {0}".format(_count))
                    item_present = False
                    detected_annotations.append(assigned_annotation)
                    for box in filter_overlapping_box:
                        if (d[0] == box[0] and d[1] == box[1] and d[2] == box[2] and d[3] == box[3] and d[4] == box[4]):
                            item_present = True
                            break

                    if not item_present:
                        filter_overlapping_box.append(
                            [int(d[0]), int(d[1]), int(d[2]), int(d[3]), d[4], int((d[0] + d[2]) / 2),
                             int((d[1] + d[3]) / 2), occupancy_flag, 0, 0])

        return filter_overlapping_box

    def writeToJSONFile(self, path, fileName, data):
        filePathNameWExt = path + fileName + '.json'
        with open(filePathNameWExt, 'w') as fp:
            json.dump(data, fp)

    def crop(self, image_path, coords, saved_location):
        image_obj = Image.open(image_path)
        cropped_image = image_obj.crop(coords)
        cropped_image.save(saved_location)


if __name__ == '__main__':
    pred = Predict('config.json','data\\test.jpg','data\\output\\')
    pred.predict()
