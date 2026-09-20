import torch 


def load_quantized_model(fp32_state_dict, q_model):

    q_model_state_dict = {} 

    for key, value in fp32_state_dict.items():
        if "Wqkv.weight" in key:

            Wq, Wk, Wv = torch.chunk(value, 3, dim = 0)
            q_model_state_dict[key.replace("Wqkv.weight", "Wq.weight")] = Wq 
            q_model_state_dict[key.replace("Wqkv.weight", "Wk.weight")] = Wk 
            q_model_state_dict[key.replace("Wqkv.weight", "Wv.weight")] = Wv


        elif "Wqkv.bias" in key: 
            bq, bk ,bv = torch.chunk(value,3, dim=0)
            q_model_state_dict[key.replace("Wqkv.bias", "Wq.bias")] = bq
            q_model_state_dict[key.replace("Wqkv.bias", "Wk.bias")] = bk
            q_model_state_dict[key.replace("Wqkv.bias", "Wv.bias")] = bv 

        elif "fc1.weight" in key:
            Wup, Wgate = torch.chunk(value,2, dim=0)
            q_model_state_dict[key.replace("fc1.weight", "Wup.weight")] = Wup 
            q_model_state_dict[key.replace("fc1.weight", "Wgate.weight")] = Wgate

        elif "fc1.bias" in key:
            bup, bgate = torch.chunk(value, 2, dim=0)
            q_model_state_dict[key.replace("fc1.bias", "Wup.bias")] = bup 
            q_model_state_dict[key.replace("fc1.bias", "Wgate.bias")] = bgate

        elif "fc2.weight" in key:
            q_model_state_dict[key.replace("fc2.weight", "Wdown.weight")] = value


        elif "fc2.bias" in key:
            q_model_state_dict[key.replace("fc2.bias", "Wdown.bias")] = value

        elif "norm1.weight" in key: 
            q_model_state_dict[key.replace("norm1.weight", "norm1.g_weights")] = value 

        elif "norm2.weight" in key:
            q_model_state_dict[key.replace("norm2.weight", "norm2.g_weights")] = value 


        else: 
            q_model_state_dict[key] = value 

    return q_model.load_state_dict(q_model_state_dict, strict= False)
