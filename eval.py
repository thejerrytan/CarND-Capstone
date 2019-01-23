import numpy as np
import tensorflow as tf
import os, datetime
from PIL import Image, ImageTk 

MODEL = {
    'ssd_inception_v2_coco_2018_sim' : 'ssd_inception_v2_coco_2018_01_28_simulator',
    'ssd_inception_v2_coco_2017_sim' : 'ssd_inception_v2_coco_2017_11_17_simulator',
}


def class_to_label(predicted_cls):
	if predicted_cls == 1:
		return 'green'
	elif predicted_cls == 2:
		return 'red'
	elif predicted_cls == 3:
		return 'yellow'
	else:
		return 'unknown'

def classify(sess, graph, image, image_tensor, boxes, scores, classes, num_detections):
    with graph.as_default():
        img_expand = np.expand_dims(image, axis=0)
        start = datetime.datetime.now()
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_expand})
        end = datetime.datetime.now()
        c = end - start

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes).astype(np.int32)

    print('Time: %.4f, SCORES: %.6f, CLASSES: %s' % (c.total_seconds(), scores[0], class_to_label(classes[0])))

    if scores[0] > 0.5:
        if classes[0] == 1:
            print('GREEN')
            return 1
        elif classes[0] == 2:
            print('RED')
            return 2
        elif classes[0] == 3:
            print('YELLOW')
            return 3
    return 4

def main():
	PATH_TO_GRAPH = r'ros/src/tl_detector/light_classification/models/%s/frozen_inference_graph.pb' % (MODEL['ssd_inception_v2_coco_2017_sim'])
	PATH_TO_IMGS = r'data/dataset-sdcnd-capstone/data/sim_training_data/sim_data_capture'

	graph = tf.Graph()
	threshold = 0.5

	image_tensor = None
	boxes = None
	scores = None
	classes = None

	with graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
			od_graph_def.ParseFromString(fid.read())
			tf.import_graph_def(od_graph_def, name='')

			image_tensor = graph.get_tensor_by_name('image_tensor:0')
			boxes = graph.get_tensor_by_name('detection_boxes:0')
			scores = graph.get_tensor_by_name('detection_scores:0')
			classes = graph.get_tensor_by_name('detection_classes:0')
			num_detections = graph.get_tensor_by_name('num_detections:0')

	sess = tf.Session(graph=graph)

	for img_filename in os.listdir(PATH_TO_IMGS):
		img = Image.open(PATH_TO_IMGS + '/%s' % (img_filename))
		result = classify(sess, graph, img, image_tensor, boxes, scores, classes, num_detections)
		img.show()


if __name__ == "__main__":
	main()
