from torch import nn
from .CLOVAS import CLOVAS
from prompt_learners.AnomalyCLIP_prompt_learner import AnomalyCLIP_PromptLearner_without_tpt
from dataset import global_defect_classes,local2global_id_map
def build_model(name: str, state_dict: dict,configs:dict):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    exclude_key = configs["exclude_key"]
    out_indices = configs["out_indices"]
    training = configs["training"]
    prompt_generator = configs["prompt_generator"] if "prompt_generator" in configs else None
    decoder = configs["decoder"]
    remove_background=configs["remove_background"] if "remove_background" in configs else None
    vpt_settings = {
        "num_tokens": configs["vpt_settings"]["num_tokens"], 
        "prompt_dim": vision_width,
        "total_d_layer": vision_layers - 1  # 11
    } if configs["vpt_mode"] else None
    tpt_settings = configs["tpt_settings"] if configs["tpt_mode"] else None
    loss_config=configs["loss_config"]
    cocoop_mode=configs["cocoop_mode"] if "cocoop_mode" in configs else None
    model = CLOVAS(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        exclude_key=exclude_key, training=training,
        vpt_settings=vpt_settings, out_indices=out_indices, tpt_settings=tpt_settings, dataset=configs["dataset"], image_size=configs["image_size"],
        decoder=decoder, prompt_generator=prompt_generator,remove_background=remove_background,cocoop_mode=cocoop_mode,use_lepe=True,use_tcs=True,use_hfca=True,loss_config=loss_config
    )
    
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    model.init_weights(state_dict)
    return model
