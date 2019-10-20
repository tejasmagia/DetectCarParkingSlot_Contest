#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs, get_yolo_boxes_v1, preprocess_input_v1, preprocess_input
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
from darkflow.net.build import TFNet

def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.85, 0.25

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes 
    ###############################
    if 'webcam' in input_path: # do detection on the first webcam
        video_reader = cv2.VideoCapture(0)

        # the main loop
        batch_size  = 1
        images      = []
        while True:
            ret_val, image = video_reader.read()
            if ret_val == True: images += [image]

            if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                for i in range(len(images)):
                    draw_boxes(images[i], batch_boxes[i], '1', obj_thresh) 
                    cv2.imshow('video with bboxes', images[i])
                images = []
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()        
    elif input_path[-4:] == '.mp4': # do detection on a video  
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

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
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

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
        #for image_path in image_paths:
        predicted_image_path = 'C:\\Users\\tejas magia\\OneDrive\\Documents\\Personal\\DataScience\\DataScience' + \
            '\\Parking Lot service\\Car-Counting-Challenge-master\\keras-yolo3-master\\input\\0_Rainy.jpg'
        predicted_image = cv2.imread(predicted_image_path)

        # predict the bounding boxes
        predicted_parking_boxes = get_yolo_boxes(infer_model, [predicted_image], net_h, net_w,
                                                 config['model']['anchors'], obj_thresh, nms_thresh)[0]

        full_parking_image = cv2.imread('C:\\Users\\tejas magia\\OneDrive\\Documents\\Personal\\DataScience\\DataScience' + \
            '\\Parking Lot service\\Car-Counting-Challenge-master\\keras-yolo3-master\\input1\\8_Rainy.jpg')
        # predict the bounding boxes
        full_parking_boxes = get_yolo_boxes(infer_model, [full_parking_image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

        crop_images = []
        full_parking_boxes_cord = []
        for box in full_parking_boxes:
            tlx = box.xmin
            tly = box.ymin
            brx = box.xmax
            bry = box.ymax
            croped_predicted_image = predicted_image[tly:bry, tlx:brx]
            full_parking_boxes_cord.append([tlx, tly, brx, bry])
            crop_images.append(croped_predicted_image)

        options = {
            'model': 'cfg/yolo.cfg',
            'load': 'yolo.weights',
            'threshold': 0.2,
            'gpu': 2.0
        }
        # Using DrawFlow model predict individual cropped slot contain Car or empty space
        tfnet = TFNet(options)

        Green_Box = []
        Red_Box = []
        for i in range(len(crop_images)):
            results = tfnet.return_predict(crop_images[i])
            for j in range(len(results)):
                if results[j]['label'] == 'car':
                    Green_Box.append(full_parking_boxes_cord[i])
        Red_Box = [cords for cords in full_parking_boxes_cord if cords not in Green_Box]

        for cords in Green_Box:
            tl = (cords[0], cords[1])
            br = (cords[2], cords[3])
            predicted_image = cv2.rectangle(predicted_image, tl, br, (0, 255, 0), 2)

        for cords in Red_Box:
            tl = (cords[0], cords[1])
            br = (cords[2], cords[3])
            predicted_image = cv2.rectangle(predicted_image, tl, br, (255, 0, 0), 1)

        cv2.imwrite('output\\output_predicted_' + predicted_image_path.split('\\')[-1], np.uint8(predicted_image))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output\\', help='path to output directory')
    
    args = argparser.parse_args()
    _main_(args)
