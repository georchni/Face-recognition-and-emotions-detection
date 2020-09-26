from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
​
emotions = ['neutral', 'happy', 'sad', 'surprised', 'angry']
​
def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. 'cam' for capturing video stream from camera", required=True,
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Demo "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                        default=0.5, type=float)
​
    return parser
​
​
def main():
    # line for log configuration
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    # parser for the arguments
    args = build_argparser().parse_args()
    # get xml model argument
    model_xml = args.model
    model2_xml = "C:\Intel\computer_vision_sdk_2018.4.420\deployment_tools\intel_models\emotions-recognition-retail-0003\FP32\emotions-recognition-retail-0003.xml"
    #model2_xml = "C:\Intel\computer_vision_sdk_2018.4.420\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml"
    #model2_xml = "C:\Intel\computer_vision_sdk_2018.4.420\deployment_tools\intel_models/age-gender-recognition-retail-0013\FP32/age-gender-recognition-retail-0013.xml"
    #model2_xml = "C:\Intel\computer_vision_sdk_2018.4.420\deployment_tools\intel_models/facial-landmarks-35-adas-0001\FP32/facial-landmarks-35-adas-0001.xml"
    # get weight model argument
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    model2_bin = os.path.splitext(model2_xml)[0] + ".bin"
    # Hardware plugin initialization for specified device and
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    # load extensions library if specified
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read intermediate representation of the model
    log.info("Reading IR...")
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    net2 = IENetwork.from_ir(model=model2_xml, weights=model2_bin)
    # check if the model is supported
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    # check if the input and output of model is the right format, here we expect just one input (one image) and one output type (bounding boxes)
    assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"
    assert len(net.outputs) <= 3, "Demo supports only single output topologies"
    # start the iterator on the input nodes
    input_blob = next(iter(net.inputs))
    input_blob2 = next(iter(net2.inputs))
    # start the iterator on the output
​
    out_blob = next(iter(net.outputs))
    out_blob2 = next(iter(net2.outputs))
    log.info("Loading IR to the plugin...")
    # load the network
    exec_net = plugin.load(network=net)
    exec_net2 = plugin.load(network=net2)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    n2, c2, h2, w2 = net2.inputs[input_blob2].shape
    del net
    del net2
    # take care of the input data (camera or video file)
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    # take care of the labels
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
​
    # opencv function to take care of the video reading/capture
    cap = cv2.VideoCapture(input_stream)
​
    log.info("Starting inference ")
    log.info("To stop the demo execution press Esc button")
​
    render_time = 0
    # open the camera
    ret, frame = cap.read()
    # if open, we loop over the incoming frames
    while cap.isOpened():
        # we get the frame
        ret, frame = cap.read()
        if not ret:
            break
        # get frame size
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        # start the time counter
        inf_start = time.time()
        # reshape the frame size and channels order to fit the model input
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
​
        # start the inference
        exec_net.infer(inputs={input_blob: in_frame})
​
        # stop the clock
        inf_end = time.time()
        det_time = inf_end - inf_start
​
        # Parse detection results
        res = exec_net.requests[0].outputs[out_blob]
​
        found_people = 0
        found_cars = 0
​
        # find all faces
        for obj in res[0][0]:
        # check the object probability , if higher than threshold it will create the bounding box
            if obj[2] > args.prob_threshold:
                found_people += 1
​
                # class ID
                class_id = int(obj[1])
                # define top left corner column value
                xmin = int(obj[3] * initial_w)
                # define top left corner row value
                ymin = int(obj[4] * initial_h)
                # define bottom right  corner column value
                xmax = int(obj[5] * initial_w)
                # define bottom right corner row value
                ymax = int(obj[6] * initial_h)
​
                # Draw box and label and class_id
                color = (0, 0, 250)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                det_label = labels_map[class_id] if labels_map else str(class_id)
                #cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                #            cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
​
                # check face emotion
                crop_frame = frame[ymin:ymax, xmin:xmax]
​
                try:
                    in_frame2 = cv2.resize(crop_frame, (w2, h2))
                except:
                    continue
                in_frame2 = in_frame2.transpose((2, 0, 1))
                in_frame2 = in_frame2.reshape((n2, c2, h2, w2))
​
                exec_net2.infer(inputs={input_blob2: in_frame2})
​
                res_emotion = exec_net2.requests[0].outputs[out_blob2]
                index = np.argmax(res_emotion[0])
​
                # blur the face if the emotion is 'sad'
                if index == 2:
                    height, width = crop_frame.shape[:2]
                    crop_frame = cv2.blur(crop_frame, (int(width / 2), int(height / 2)))
                    frame[ymin:ymax, xmin:xmax] = crop_frame
​
        # label = "%s" % (emotions[index])
        # cv2.putText(frame, label, (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 230), 2)
​
        # show the result image
        frame = cv2.resize(frame, (int(w*2.5), int(h*2)))
        cv2.imshow("Detection Results", frame)
​
        key = cv2.waitKey(1)
        if key == 27:
            break
​
    cv2.destroyAllWindows()
    del exec_net
    del plugin
​
if __name__ == '__main__':
    sys.exit(main() or 0)
