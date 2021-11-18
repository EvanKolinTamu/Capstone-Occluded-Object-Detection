import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

already_detected = []
#count_I = int(input("how many 'I' shapes do you want?"))

count_L = int(input("how many 'L' shapes do you want?"))
count_L2 = count_L
#count_T = int(input("how many 'T' shapes do you want?"))
#count_10 = int(input("how many '10' shapes do you want?"))

ilt_array = [0,1,2,10]

print("Starting streaming")
pipeline.start(config)

# load tensorflow
print("[INFO] Loading model...")
PATH_TO_CKPT = "frozen_inference_graph.pb"

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#image_tensor = 'image_tensor:0'
#detection_boxes = 'detection_boxes:0'
#detection_scores = 'detection_scores:0'
#detection_classes = 'detection_classes:0'
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#num_detections = 'num_detections:0'
print("[INFO] Model loaded.")
colors_hash = {}


should_we_loop = True
while should_we_loop:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth = frames.get_depth_frame()

    count_L2 = count_L


    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    scaled_size = (color_frame.width, color_frame.height)
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image_expanded = np.expand_dims(color_image, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                feed_dict={image_tensor: image_expanded})

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    for idx in range(int(num)):
        class_ = classes[idx]
        score = scores[idx]
        box = boxes[idx]

        #count_L2 = count_L

        if class_ not in colors_hash:
            colors_hash[class_] = (0, 0, 255)
        
        if score > 0.8:
            left = int(box[1] * color_frame.width)
            top = int(box[0] * color_frame.height)
            right = int(box[3] * color_frame.width)
            bottom = int(box[2] * color_frame.height)

            xcoord = (left+right)/2
            ycoord = (top+bottom)/2
            
            p1 = (left, top)
            p2 = (right, bottom)

            #distance text
            dist = depth.get_distance(int(xcoord), int(ycoord))
            dist2 = '{0:0.3g}'.format(dist)
            dist_txt = str(dist2) + "[m]"

            
            if (class_ in ilt_array) and (count_L2 > 0):
                box_width=6
                fontScale = 1.3
                colors_hash[class_] = (0, 255, 0)
                count_L2 = count_L2 -1
                print(class_, " is at: ", left, right, top, bottom)

            else:
                box_width=2
                fontScale=1

            print("count L2",count_L2)
            print("count L",count_L)


            #color
            r, g, b = colors_hash[class_]
            

            cv2.rectangle(color_image, p1, p2, (int(r), int(g), int(b)), box_width, 1)

            classes_txt = str(class_)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (p1[0], p1[1] + 20)
            #fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2
            cv2.putText(color_image, classes_txt,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            #label distance
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (p1[0], p1[1] + 40)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2
            cv2.putText(color_image, dist_txt,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)


    if count_L2 > 0:
        print("User has requested ", count_L, " pieces, but we have ", count_L - count_L2, " pieces.")
        print("(remove later) You need remove the selected pieces, put them in the organized bin and re-image the remaining pieces please.")
    if count_L2 == 0:

        should_we_loop = False
        print("Order completed")

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    cv2.waitKey(1)

print("[INFO] stop streaming ...")
pipeline.stop()
