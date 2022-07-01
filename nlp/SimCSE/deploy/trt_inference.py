from transformers import BertTokenizer
import tensorrt as trt
import common1
import time
import transformers
import torch

"""
a、获取 engine，建立上下文
"""
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

engine_model_path = "bert_dynamic-fp16.engine"
# Build a TensorRT engine.
engine = get_engine(engine_model_path)
# Contexts are used to perform inference.
context = engine.create_execution_context()


"""
b、从engine中获取inputs, outputs, bindings, stream 的格式以及分配缓存
"""
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


tokenizer = transformers.AutoTokenizer.from_pretrained("../chinese-bert-wwm-ext", do_lower_case=True)
text = ["中餐厨师","后厨","厨师长"]
# text = ["中餐厨师","中餐厨师","中餐厨师"]
inputs = tokenizer(
    text,
    None,
    truncation=True,
    add_special_tokens=True,
    max_length=16,
    padding='max_length'
)
ids = torch.tensor(inputs['input_ids'], dtype=torch.long).to("cuda").unsqueeze(0)
mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).to("cuda").unsqueeze(0)
token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).to("cuda").unsqueeze(0)

tokens_id = to_numpy(ids.int())
token_type_ids = to_numpy(token_type_ids.int())
attention_mask = to_numpy(mask.int())

context.active_optimization_profile = 0
origin_inputshape = context.get_binding_shape(0)
origin_inputshape[0], origin_inputshape[1],origin_inputshape[2] = tokens_id.shape
context.set_binding_shape(0, (origin_inputshape))
context.set_binding_shape(1, (origin_inputshape))
context.set_binding_shape(2, (origin_inputshape))

"""
c、输入数据填充
"""
inputs, outputs, bindings, stream = common1.allocate_buffers_v2(engine, context)
inputs[0].host = tokens_id
inputs[1].host = token_type_ids
inputs[2].host = attention_mask

"""
d、tensorrt推理
"""
inf_time = 0
for i in range(100):
    start = time.time()
    trt_outputs = common1.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    end = time.time()
    inf_time+=(end-start)
print(f"time is {(inf_time)*1000/100} ms")

from scipy.spatial.distance import cosine
a = trt_outputs[0][0:768]
b = trt_outputs[0][768:768*2]
c = trt_outputs[0][768*2:]
cosine_sim_0_1 = 1 - cosine(a, b)
cosine_sim_0_2 = 1 - cosine(a, c)
print(cosine_sim_0_1)
print(cosine_sim_0_2)
