import os

def get_data(args, image_processor, tokenizer, dataset_type, epoch=0):
    if "keyframe" in dataset_type:
        from .keyframe_data import get_multi_dataset
        return get_multi_dataset(args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer)
    else:
        from .multi_cam_data import get_multi_dataset
        return get_multi_dataset(args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer)
