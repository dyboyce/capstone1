#================================================================
#
#   File name   : object_tracker.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : code to track detected object from video or webcam
#
#================================================================
import os

from azure.cosmos import PartitionKey

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time
import pandas as pd
import json
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as errors
import azure.cosmos.documents as documents
import azure.cosmos.http_constants as http_constants

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

video_path   = "./IMAGES/test.mp4"

def Object_tracking(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = []):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2 = [], []

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    video_id = output_path
    df = pd.DataFrame()
    frame_id = 0

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())
    while True:
        _, frame = vid.read()
        frame_id += 1

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        #image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        #t1 = time.time()
        #pred_bbox = Yolo.predict(image_data)
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        classed_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            #changed '...+ [tracking_id, index])' to class name instead
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) #f Structure data, that we could use it with our draw_bbox function
            classed_bboxes.append(bbox.tolist() + [tracking_id, class_name])
        #add frame id and class instead of index TODO
        #print(tracked_bboxes)
        #we can export ids/classes from here - or extract from tracker obj
        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=CLASSES, tracking=True)
        vid_details = [video_id, frame_id, classed_bboxes]
        #print(vid_details)

        df = df.append(pd.Series(vid_details, index=["Video_id", "Frame", "Boxes"]), ignore_index=True)
        #print(df)
        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        
        image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)
        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
        if frame_id >30:
            break

    cv2.destroyAllWindows()
    print(df)
    return df

yolo = Load_Yolo_model()
df2 = Object_tracking(yolo, video_path, "detection.avi", input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = ["person"])

# Initialize the Cosmos client

config = {
    "endpoint": "https://dylansql2test.documents.azure.com:443/",
    "primarykey": "T0oS8Kgz9pN0Mt8LAtHdxnied9EwFDwE485ZAaJ7gLlfi7N9ua17VEtZcg13wN42jXjBPFsy6eUCb0UPgvR9dQ=="
}

# Create the cosmos client
client = cosmos_client.CosmosClient(config["endpoint"], config["primarykey"])

# Create a database
# https://docs.microsoft.com/en-us/python/api/azure-cosmos/azure.cosmos.cosmos_client.cosmosclient?view=azure-python#createdatabase-database--options-none-

database_name = 'dylantestdb'
container_name = 'testcont3'
try:
    database = client.create_database_if_not_exists(database_name)
except (errors.CosmosHttpResponseError, errors.CosmosResourceExistsError) as e:
    database = client.get_database_client(database_name)


# Create a collection of items in a Container
# Items are the individual rows/records of your dataset
# https://docs.microsoft.com/en-us/python/api/azure-cosmos/azure.cosmos.cosmos_client.cosmosclient?view=azure-python#createcontainer-database-link--collection--options-none-

#database_link = 'dbs/' + 'dylantestdb'
"""container_definition = {'id': 'dylanTcontainer',
    #                    'partitionKey':
    #                                {
       #                                 'paths': ['/video_id'],
         #                               'kind': documents.PartitionKind.Hash
                                    }
                        }"""
database = client.get_database_client(database_name)
partition_key = PartitionKey(path='/video_id', kind='Hash')
try:
    container = database.create_container_if_not_exists(container_name, partition_key)
except errors.CosmosResourceExistsError as e:
    print('A container with id \'{0}\' already exists'.format(id))
    container = database.get_container_client(container_name)

"""
# Download and read csv file
df2 = pd.read_csv('https://globaldatalab.org/assets/2019/09/SHDI%20Complete%203.0.csv',encoding='ISO-8859–1',dtype='str')
# Reset index - creates a column called 'index'
df2 = df2.reset_index()
# Rename that new column 'id'
# Cosmos DB needs one column named 'id'.
df2 = df2.rename(columns={'index':'id'})
# Convert the id column to a string - this is a document database.
df2['id'] = df2['id'].astype(str)
"""
# https://docs.microsoft.com/en-us/python/api/azure-cosmos/azure.cosmos.cosmos_client.cosmosclient?view=azure-python#upsertitem-database-or-container-link--document--options-none-
# Write rows of a pandas DataFrame as items to the Database Container

# Create Connection Link string
#collection_link = database_link + '/colls/' + 'HDIcontainer'

df2 = df2.reset_index()
df2 = df2.rename(columns={'index':'id'})
df2['id'] = df2['id'].astype(str)

# Write rows of a pandas DataFrame as items to the Database Container
for i in range(0,df2.shape[0]):
    # create a dictionary for the selected row
    data_dict = dict(df2.iloc[i,:])
    # convert the dictionary to a json object.
    data_dict = json.dumps(data_dict)
    insert_data = container.upsert_item(body=json.loads(data_dict))
print('Records inserted successfully.')