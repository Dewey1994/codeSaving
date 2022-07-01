import torch
import transformers

from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# dummy_input = tokenizer(text, return_tensors='pt')
#
# dummy_input = (dummy_input['input_ids'], dummy_input['token_type_ids'], dummy_input['attention_mask'])
#
# torch.onnx.export(model, dummy_input, "./bert.onnx", opset_version=10)


def get_model():
    model = torch.load('../model.pth', map_location="cuda")
    return model



def get_onnx(model,onnx_save_path,input_ids,token_type_ids,attention_mask,batch_size=1,num_sent=3):

    _ = torch.onnx.export(model, (input_ids,token_type_ids,attention_mask),
                          onnx_save_path,
                          verbose=True,
                          do_constant_folding=True,
                          opset_version=11,
                          input_names=['input_ids', 'token_type_ids','attention_mask'],
                          output_names=["output"],
                          dynamic_axes={'input_ids':[2],
                                        'token_type_ids':[2],
                                        'attention_mask':[2]}
                          )


if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained("../chinese-bert-wwm-ext", do_lower_case=True)
    text = ["中餐厨师","中餐厨师","中餐厨师"]
    inputs = tokenizer(
        text,
        None,
        truncation=True,
        add_special_tokens=True,
        max_length=20,
        padding='max_length'
    )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).to("cuda").unsqueeze(0)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).to("cuda").unsqueeze(0)
    token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).to("cuda").unsqueeze(0)
    model = get_model()
    get_onnx(model,"model-default.onnx",ids,token_type_ids,mask)