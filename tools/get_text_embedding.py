"""
Precompute embeddings of instructions.
"""
import re
import json
from pathlib import Path
import itertools
from typing import List, Tuple, Literal, Dict, Optional
import pickle

import tap
import transformers
from tqdm.auto import tqdm
import torch


Annotations = Dict[str, Dict[int, List[str]]]
TextEncoder = Literal["bert", "clip"]


class Arguments(tap.Tap):
    tasks: Tuple[str, ...] = [x for x in "place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap".split(" ")]
    output: Path = "instructions.pkl"
    batch_size: int = 10
    encoder: TextEncoder = "clip"
    model_max_length: int = 53
    variations: Tuple[int, ...] = list(range(199))
    device: str = "cuda"
    annotations: Tuple[Path, ...] = ()
    zero: bool = False
    verbose: bool = False


def parse_int(s):
    return int(re.findall(r"\d+", s)[0])


def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("/project/robotic/modelzoo/openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model


def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "/project/robotic/modelzoo/openai/clip-vit-base-patch32"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer


def load_annotations(annotations: Tuple[Path, ...]) -> Annotations:
    data = []
    for annotation in annotations:
        with open(annotation) as fid:
            data += json.load(fid)

    items: Annotations = {}
    for item in data:
        task = item["fields"]["task"]
        variation = item["fields"]["variation"]
        instruction = item["fields"]["instruction"]

        if instruction == "":
            continue

        if task not in items:
            items[task] = {}

        if variation not in items[task]:
            items[task][variation] = []

        items[task][variation].append(instruction)

    # merge annotations for push_buttonsX (same variations)
    push_buttons = ("push_buttons", "push_buttons3")
    for task, task2 in itertools.product(push_buttons, push_buttons):
        items[task] = items.get(task, {})
        for variation, instrs in items.get(task2, {}).items():
            items[task][variation] = instrs + items[task].get(variation, [])

    # display statistics
    for task, values in items.items():
        print(task, ":", sorted(values.keys()))

    return items


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    output_file = '/project/robotic/robo_mm/data_folder/robomimic_lang_clip.pkl'
    with open('/project/robotic/robo_mm/data_folder/robomimic_lang.json') as fid:
        annotations = json.load(fid)

    tokenizer = load_tokenizer(args.encoder)
    tokenizer.model_max_length = args.model_max_length

    model = load_model(args.encoder)
    model = model.to(args.device)

    instructions = []
    for instr in tqdm(annotations):
    
        tokens = tokenizer([instr], padding="max_length")["input_ids"]
        lengths = [len(t) for t in tokens]
        if any(l > args.model_max_length for l in lengths):
            raise RuntimeError(f"Too long instructions: {lengths}")

        tokens = torch.tensor(tokens).to(args.device)
        with torch.no_grad():
            pred = model(tokens).last_hidden_state
        pred = pred.cpu()
        instructions.append(pred)

    with open(output_file, "wb") as f:
        pickle.dump(instructions, f)


with open('/project/robotic/robo_mm/data_folder/calvin_lang_clip.pkl', 'rb') as file:
    # 加载对象
    calvin_lang_clip = pickle.load(file)
with open('/project/robotic/robo_mm/data_folder/metaworld_lang_clip.pkl', 'rb') as file:
    # 加载对象
    metaworld_lang_clip = pickle.load(file)
with open('/project/robotic/robo_mm/data_folder/libero_lang_clip.pkl', 'rb') as file:
    # 加载对象
    libero_lang_clip = pickle.load(file)
with open('/project/robotic/robo_mm/data_folder/robocasa_lang_clip.pkl', 'rb') as file:
    # 加载对象
    robocasa_lang_clip = pickle.load(file)
with open('/project/robotic/robo_mm/data_folder/robomimic_lang_clip.pkl', 'rb') as file:
    # 加载对象
    robomimic_lang_clip = pickle.load(file)

calvin_metaworld_lang_clip = torch.vstack(calvin_lang_clip + metaworld_lang_clip)
libero_lang_clip = torch.vstack(libero_lang_clip)
robocasa_lang_clip = torch.vstack(robocasa_lang_clip)
robomimic_lang_clip = torch.vstack(robomimic_lang_clip)

calvin_metaworld_lang_clip = calvin_metaworld_lang_clip.reshape(-1, 53*512)
calvin_metaworld_lang_clip_norm = calvin_metaworld_lang_clip / calvin_metaworld_lang_clip.norm(p=2, dim=1, keepdim=True)
libero_lang_clip = libero_lang_clip.reshape(-1, 53*512)
libero_lang_clip_norm = libero_lang_clip / libero_lang_clip.norm(p=2, dim=1, keepdim=True)
robocasa_lang_clip = robocasa_lang_clip.reshape(-1, 53*512)
robocasa_lang_clip_norm = robocasa_lang_clip / robocasa_lang_clip.norm(p=2, dim=1, keepdim=True)
robomimic_lang_clip = robomimic_lang_clip.reshape(-1, 53*512)
robomimic_lang_clip_norm = robomimic_lang_clip / robomimic_lang_clip.norm(p=2, dim=1, keepdim=True)

libero_m = torch.mm(libero_lang_clip_norm, calvin_metaworld_lang_clip_norm.t())
(libero_m/libero_m.sum(dim=1, keepdim=True)).max(-1)[0].mean()

robocasa_m = torch.mm(robocasa_lang_clip_norm, calvin_metaworld_lang_clip_norm.t())
(robocasa_m/robocasa_m.sum(dim=1, keepdim=True)).max(-1)[0].mean()

robomimic_m = torch.mm(robomimic_lang_clip_norm, calvin_metaworld_lang_clip_norm.t())
(robomimic_m/robomimic_m.sum(dim=1, keepdim=True)).max(-1)[0].mean()