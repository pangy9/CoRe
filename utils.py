from modules.my_clip.clip_model import CLIPTextModel
from transformers import CLIPTokenizer
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import random
import torch.nn.functional as F
@torch.no_grad()
def get_token_embeds(
    tokens : str|list[str],
    tokenizer : CLIPTokenizer,
    text_encoder: CLIPTextModel,
):
    if isinstance(tokens,list):
        tokens=' '.join(tokens)
    
    token_ids=tokenizer(
        tokens,
        padding="do_not_pad",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(text_encoder.device) # (1,k)
    embeds=text_encoder.get_input_embeddings().weight.data[token_ids[0]] # (k+2,1024)
    return embeds[1:-1] #(k,1024)


@torch.no_grad()
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def freeze_model(model:torch.nn.Module,to_freeze:list[str]):
    for name,params in model.named_parameters():
        for freeze_name in to_freeze:
            if freeze_name in name:
                params.requires_grad_(False)

def unfreeze_model(model:torch.nn.Module,to_unfreeze:list[str]):
    for name,params in model.named_parameters():
        for unfreeze_name in to_unfreeze:
            if unfreeze_name in name:
                params.requires_grad_(True)

def get_model_params(model:torch.nn.Module,names:list[str]):
    params_list=[]
    for model_name,params in model.named_parameters():
        for name in names:
            if name in model_name:
                params_list.append(params)
    return params_list

def get_context_embedding_loss(
    object_tokens, 
    category_tokens, 
    training_embedding,
    tokenizer, 
    text_encoder,
    random_prompt, 
    prompts_cosine_weight,
    weight_dtype,
):
    '''calculate context_embedding_loss
        and return it with object_prompt_embedding, category_prompt_embedding, start_index for S* and  eot_index
    '''
    context_embedding_loss = 0
    object_prompt = random_prompt.format(' '.join(object_tokens))
    category_prompt = random_prompt.format(' '.join(category_tokens))
    
    category_token_ids=tokenizer(
        category_tokens,
        padding="do_not_pad",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0] # (1,k)
    category_token_ids = category_token_ids[(category_token_ids != 49407) & (category_token_ids != 49406)]
    
    if len(category_token_ids)==0:
        raise ValueError("Category token is not valid, it will be interpreted as <eot> or <sot>.")
    
    object_prompt_ids = tokenizer(
        object_prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]
    category_prompt_ids = tokenizer(  
        category_prompt,  
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]
    start_index = (object_prompt_ids == 49408).nonzero(as_tuple=True)[0].item()
    eot_index = (object_prompt_ids == 49407).nonzero(as_tuple=True)[0].item()

    object_prompt_embedding = text_encoder.get_input_embeddings().weight.data[object_prompt_ids]
    object_prompt_embedding[start_index:start_index + len(object_tokens)] = training_embedding
    object_prompt_embedding = object_prompt_embedding.unsqueeze(0)

    category_prompt_embedding = text_encoder.get_input_embeddings().weight.data[category_prompt_ids]
    category_prompt_embedding = category_prompt_embedding.unsqueeze(0)
    
    if prompts_cosine_weight != 0:
        object_prompt_condition = text_encoder(inputs_embeds=object_prompt_embedding)[0].to(dtype=weight_dtype)
        category_prompt_condition = text_encoder(inputs_embeds=category_prompt_embedding)[0].to(dtype=weight_dtype)

        cond_for_loss_object = torch.cat([object_prompt_condition[:, :start_index], object_prompt_condition[:, start_index + len(object_tokens): eot_index + 1]], dim=1)
        cond_for_loss_category = torch.cat([category_prompt_condition[:, :start_index], category_prompt_condition[:, start_index + len(category_token_ids): eot_index + 1]], dim=1)

        prompts_cosine_loss = 1 - F.cosine_similarity(cond_for_loss_category, cond_for_loss_object, dim=1).mean()
        prompts_cosine_loss *= prompts_cosine_weight

        context_embedding_loss += prompts_cosine_loss

    output = {
        "context_embedding_loss": context_embedding_loss,
        "object_prompt_embedding": object_prompt_embedding,
        "category_prompt_embedding": category_prompt_embedding,
        "start_index": start_index,
        "eot_index": eot_index,
        "category_token_ids_len": len(category_token_ids), # Used for slicing and shifting to the appropriate position.
    }
    return output
