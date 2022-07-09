import tensorrt as trt
import common
import os
import argparse
from transformers import BertTokenizer
import transformers

logger = trt.Logger(trt.Logger.WARNING)
tokenizer = transformers.AutoTokenizer.from_pretrained("../chinese-bert-wwm-ext", do_lower_case=True)
text = ["中餐厨师","中餐厨师","中餐厨师"]
# inputs = tokenizer(
#     text,
#     None,
#     truncation=True,
#     add_special_tokens=True,
#     max_length=20,
#     padding='max_length'
# )



def get_engine_static(args, calib):
    def build_engine():
        with trt.Builder(logger) as builder, builder.create_network(
                common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, logger) as parser, trt.Runtime(logger) as runtime:
            config.max_workspace_size = 1 << 28
            builder.max_batch_size = 1
            if not os.path.exists(args.onnx_file_path):
                print('ONNX file {} not found, please run onnx_2_trt.py first to generate it.'.format(
                    args.onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
            with open(args.onnx_file_path, 'rb') as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR:Failed to parse the ONNX file")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(args.engine_file_path, "wb") as f:
                f.write(plan)
            return engine
    if os.path.exists(args.engine_file_path):
        print("Reading engine from file {}".format(args.engine_file_path))
        with open(args.engine_file_path, 'rb') as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def get_engine_dynamic(args, calib, dynamic_shape=None):

    def build_engine(dynamic_shape=None):
        with trt.Builder(logger) as builder, builder.create_network(
                common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, logger) as parser, trt.Runtime(logger) as runtime:
            config.max_workspace_size = (1 << 30) * 2  # 2 GB
            builder.max_batch_size = 1
            if args.data_type=='fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            if not os.path.exists(args.onnx_file_path):
                print('ONNX file {} not found, please run onnx_2_trt.py first to generate it.'.format(
                    args.onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
            with open(args.onnx_file_path, 'rb') as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR:Failed to parse the ONNX file")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            if len(dynamic_shape) > 0:
                print("====>using dynamic shape")
                profile = builder.create_optimization_profile()
                for binding_name in dynamic_shapes:
                    dynamic_shape = dynamic_shapes[binding_name]
                    min_shape, opt_shape, max_shape = dynamic_shape
                    profile.set_shape(binding_name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            # engine = builder.build_engine(network,config)
            print("Completed creating Engine")
            with open(args.engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(args.engine_file_path):
        print("Reading engine from file {}".format(args.engine_file_path))
        with open(args.engine_file_path, 'rb') as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(dynamic_shape)


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", fp16_mode=True, int8_mode=False,
               save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # 这句一定需要加上不然会报错

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(logger) as builder, \
                builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, logger) as parser, \
                trt.Runtime(logger) as runtime:

            print(network.num_layers)
            print(network.num_inputs)
            print(network.num_outputs)
            print(network.name)

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 256MiB
            builder.max_batch_size = max_batch_size
            if fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            elif int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)
            else:
                config.set_flag(trt.BuilderFlag.REFIT)

            flag = builder.is_network_supported(network, config)
            print('flag', flag)

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                # print(type(model.read()))
                parser.parse(model.read())
                # parser.parse_from_file(onnx_file_path)
                assert network.num_layers > 0, 'Failed to parse ONNX model.Please check if the ONNX model is compatible '

            # last_layer = network.get_layer(network.num_layers - 1)
            # network.mark_output(last_layer.get_output(0))

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pytorch2TensorRT args")
    parser.add_argument("--batch_size", type=int, default=32, help='batch_size')
    parser.add_argument("--channel", type=int, default=3, help='input channel')
    parser.add_argument("--height", type=int, default=224, help='input height')
    parser.add_argument("--width", type=int, default=224, help='input width')
    parser.add_argument("--cache_file", type=str, default='cache', help='cache_file')
    # parser.add_argument("--mode", type=str, default='fp16', help='fp32, fp16 or int8')
    parser.add_argument("--onnx_file_path", type=str, default='model-default.onnx', help='onnx_file_path')
    parser.add_argument("--engine_file_path", type=str, default='bert_dynamic-fp16.engine', help='engine_file_path')
    parser.add_argument("--input_image_path", type=str, default='binoculars.jpeg', help='image path')
    parser.add_argument("--labels_file", type=str, default='class_labels.txt', help='label path')
    parser.add_argument("--data_type", type=str, default='fp16', help='fp16 fp32 int8')
    args = parser.parse_args()
    print(args)
    dynamic_shapes = {
        "input_ids": ((1,3, 5), (1,3,16), (1,3,16)),
        "token_type_ids": ((1, 3, 5), (1,3,16), (1,3,16)),
        "attention_mask": ((1, 3, 5), (1,3,16), (1,3,16)),
                      }
    get_engine_dynamic(args,calib=None,dynamic_shape=dynamic_shapes)
