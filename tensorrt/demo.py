# import tensorflow.compat.v1 as tf
import numpy as np
import tensorrt as trt
import common
import cv2
import argparse
#create logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
print(trt.__version__)

MAX_BATCH_SIZE = -1
BN_EPSILON = 0.0010000000474974513
BALL_CONSTANT = 9.99999993922529e-09


parser = argparse.ArgumentParser("Hi this is the TensorRT demo")
parser.add_argument('--mode', '-m', type=str, choices=['build', 'inference'])
parser.add_argument('--engine', '-e', type=str, default='../cp-9-0505.engine')
parser.add_argument('--onnx', '-o', type=str, default='../cp-9-0505.onnx')

args = parser.parse_args()

engine_path = args.engine
onnx_path  = '../cp-9-0505.onnx'


class ModelData(object):
    INPUT_NAME = 'input_1'
    OUTPUT_NAME = ['output_1', 'output_2', 'output_3', 'output_4']
    INPUT_SHAPE = (180, 320)
    DTYPE = trt.float32

def build_engine_from_onnx(onnx_path: str, max_batch_size: int):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config: 
        builder.max_batch_size = max_batch_size
        # network = builder.create_network(1 <<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(1 <<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # network.add_input("foo", trt.float32, (-1, -1, 1))
        config.max_workspace_size = common.GiB(1)
        profile = builder.create_optimization_profile()
        profile.set_shape("input_1", (1, 180, 320), (1, 180, 320), (1, 180, 320)) 
        config.add_optimization_profile(profile)
        # Populate the network using ONNX parser
        parser = trt.OnnxParser(network, TRT_LOGGER)
        is_parsed = parser.parse_from_file(onnx_path)
        assert is_parsed
        # Build and return an engine.
        return builder.build_engine(network, config)

if args.mode == 'build':
    engine = build_engine_from_onnx(onnx_path, MAX_BATCH_SIZE)
    common.serialize_engine_to_file(engine, engine_path)

elif args.mode == 'inference':
    engine_2 = common.deserialize_engine_from_file(engine_path=engine_path, logger=TRT_LOGGER)
    # print(engine_2)
    # Allocate input, output on host memory and GPU device memory
    inputs_2, outputs_2, bindings_2, stream_2 = common.allocate_buffers(engine_2) #Return list of inputs/outputs host_device_memory(buffer) devicebindingsbuffers, cuda stream
    # # Create execution context from engine
    context_2 = engine_2.create_execution_context()
    context_2.set_binding_shape(0, (1, 180, 320, 1))
    # # # Load input array to host memory
    bboxes  = np.random.rand(4, 180, 320, 1).ravel()
    np.copyto(inputs_2[0].host, bboxes)
    # # # Do inferences
    out= common.do_inference(context_2, bindings=bindings_2, inputs=inputs_2, outputs=outputs_2, stream=stream_2, batch_size=4)

    print(out)