#!/usr/bin/env python
import os, sys
import cv2
import pyrealsense2 as rs
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import math

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync



@torch.no_grad()
def run():

    weights='best.pt'  # model.pt path(s)
    imgsz=640  # inference size (pixels)
    conf_thres=0.70  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=20  # maximum detections per image
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    stride = 32
    device_num=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    update=False  # update all models
    name='exp'  # save results to project/name



    count_I = int(input("how many 'I' shapes do you want? "))
    count_L = int(input("how many 'L' shapes do you want? "))
    count_T = int(input("how many 'T' shapes do you want? "))


    # Initialize
    set_logging()
    device = select_device(device_num)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location = device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location = device)['model']).to(device).eval()

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)


    frame_counter = 0
    shake_Count = 0

    while(True):

        order_list = []
        count_I2 = count_I
        count_L2 = count_L
        count_T2 = count_T


        t0 = time.time()

        frames = pipeline.wait_for_frames()

        depth = frames.get_depth_frame()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue


        

        img = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # check for common shapes
        s = np.stack([letterbox(x, imgsz, stride=stride)[0].shape for x in img], 0)  # shapes
        rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        # Letterbox
        img0 = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=augment,
                     visualize=increment_path(save_dir / 'features', mkdir=True) if visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, img0)

        #if(frame_counter > 15):    
            
         
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            #print(Annotator(img0, line_width=line_thickness, example=str(names)))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()



                if(frame_counter > 15): 


                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        
                     
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')


                        
                        
                        
                        #Check for if each piece is still required for the order
                        #this makes sure that each individual piece is checked to be highlighted green
                        if (count_I2 != 0) and (names[c] == 'I'):
                            print("coint_I2 is: ", count_I2)
                            count_I2-= 1 
                            boundingbox_color = (0, 255, 0)
                            bBox = annotator.box_label(xyxy, label, color=boundingbox_color)
                            order_list.append("I")
                            #LowerLeft X and Y
                            #UpperRight X and Y
                            LLBBX = bBox[0][0]
                            LLBBY = bBox[0][1]
                            URBBX = bBox[1][0]
                            URBBY = bBox[1][1]
                            #Translating pixel values into real distance
                            bB1X = round(LLBBX/640*29,2)
                            bB1Y = round(LLBBY/480*22,2)
                            bB2X = round(URBBX/640*29,2)
                            bB2Y = round(URBBY/480*22,2)
                            #putting values into a readable format
                            bboxCords = [(bB1X,bB1Y),(bB2X,bB2Y),bBox[2]]
                            #append to order list
                            order_list.append(bboxCords)

                        elif (count_L2 != 0) and (names[c] == 'L'):
                            print("coint_L2 is: ", count_L2)
                            count_L2-= 1
                            boundingbox_color = (0, 255, 0)
                            bBox = annotator.box_label(xyxy, label, color=boundingbox_color)
                            order_list.append("L")
                            #LowerLeft X and Y
                            #UpperRight X and Y
                            LLBBX = bBox[0][0]
                            LLBBY = bBox[0][1]
                            URBBX = bBox[1][0]
                            URBBY = bBox[1][1]
                            #Translating pixel values into real distance
                            bB1X = round(LLBBX/640*29,2)
                            bB1Y = round(LLBBY/480*22,2)
                            bB2X = round(URBBX/640*29,2)
                            bB2Y = round(URBBY/480*22,2)
                            #putting values into a readable format
                            bboxCords = [(bB1X,bB1Y),(bB2X,bB2Y),bBox[2]]
                            #append to order list
                            order_list.append(bboxCords)
                        elif (count_T2 != 0) and (names[c] == 'T'):
                            print("coint_T2 is: ", count_T2)
                            count_T2-= 1
                            boundingbox_color = (0, 255, 0) 
                            bBox = annotator.box_label(xyxy, label, color=boundingbox_color)
                            order_list.append("T")
                            #LowerLeft X and Y
                            #UpperRight X and Y
                            LLBBX = bBox[0][0]
                            LLBBY = bBox[0][1]
                            URBBX = bBox[1][0]
                            URBBY = bBox[1][1]
                            #Translating pixel values into real distance
                            bB1X = round(LLBBX/640*29,2)
                            bB1Y = round(LLBBY/480*22,2)
                            bB2X = round(URBBX/640*29,2)
                            bB2Y = round(URBBY/480*22,2)
                            #putting values into a readable format
                            bboxCords = [(bB1X,bB1Y),(bB2X,bB2Y),bBox[2]]
                            #append to order list
                            order_list.append(bboxCords)  
                        else:
                            boundingbox_color = (0, 0, 255)
                            bBox = annotator.box_label(xyxy, label, color=boundingbox_color)

                        
                        print("this is count_I2 ",count_I2)
                        print("this is count_L2 ",count_L2)
                        print("this is count_T2 ",count_T2)
                        print("this is  names[int(c)]", names[int(c)])

                    print("SHAKE: ", shake_Count)
                    if not (count_I2 == count_L2 == count_T2 == 0) and (len(order_list) == 0):
                        if(shake_Count < 3):
                            order_list.append("Error: No desired piece found. Solution: Shake Bucket.")
                            shake_Count += 1
                        else:
                            order_list.append("The part you're looking for is out of stock.")
                    elif not (count_I2 == count_L2 == count_T2 == 0):
                        order_list.append("Order is NOT complete. Remove the parts at these locations and Re-Image.")
                    elif len(order_list)==0:
                        order_list.append("Order Complete.")
                    else:
                        order_list.append("Remove the parts at these locations and order will be complete.")



                    print("This is order list: ",order_list)  
                    input("Press Enter to continue...")     
                    count_I = count_I2
                    count_L = count_L2
                    count_T = count_T2


        #This is for the regular image   
        cv2.imshow("IMAGE", img0)
        #this is for depth to color screen 
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
        #cv2.imshow("DEPTH", depth_colormap)
           

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  


if __name__ == '__main__':
    run()