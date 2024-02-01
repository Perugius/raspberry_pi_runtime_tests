import tflite_runtime.interpreter as tflite
import onnxruntime as ort

import numpy as np
import time

inputs = np.load("test_data/rand_model_13.npy")
inputs = np.reshape(np.float32(inputs), (1, 15008, 2))
#inputs_onnx = np.float32(inputs)

def initialize_tflite(modelPath):
    interpreter = tflite.Interpreter(model_path=modelPath, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], inputs)
    return interpreter

def inference_tflite(interpreter):
    start_tflite = time.time()
    interpreter.invoke()
    end_tflite = time.time()
    return (end_tflite - start_tflite)

def initialize_onnx(modelPath):
    sess = ort.InferenceSession(modelPath)
    return sess

def inference_onnx(sess):
    start_onnx = time.time()
    #not saving results
    sess.run(None, {'input_1': inputs})
    end_onnx = time.time()
    return (end_onnx - start_onnx)

#get mean of a python array by converting to np array
def get_mean(arr):
    arr = np.array(arr)
    return np.mean(arr)

if __name__ == "__main__":
    #load interpreters with corresponding model
    interpreter = initialize_tflite("models/model_13_egc.tflite")
    #sess = initialize_onnx("models/model_13_egc.onnx")
    #initialize arrays to save inference durations to later average
    inferenceDurations_tflite = []
    inferenceDurations_onnx = []

    #run x instances of inference and save times
    for i in range(0, 5000):
        inferenceDurations_tflite.append(inference_tflite(interpreter))
        print(i)
    
    # for i in range(0, 1000):
    #     inferenceDurations_onnx.append(inference_onnx(sess))
    #     print(i)
    
    # onnx_mean = get_mean(inferenceDurations_onnx)
    # print("average inference time 5000 measurements ONNX: "+str(onnx_mean))
    tflite_mean = get_mean(inferenceDurations_tflite)
    print("average inference time 5000 measurements TFLITE: "+str(tflite_mean))



