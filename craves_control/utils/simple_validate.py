# Modified by Weichao Qiu @ 2018
import torch
from craves_control.utils.transforms import fliplr, flip_back, multi_scale_merge
from craves_control.utils.evaluation import accuracy, final_preds, final_preds_bbox, get_preds
import numpy as np

# Simplify this script
def validate(inputs, meta, model, data_dir, flip=True,
             scales=[0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6], multi_scale=False):

    inputs = inputs.unsqueeze(0)
    input_var = torch.autograd.Variable(inputs)

    with torch.no_grad():

        output = model(input_var)

        score_map = output[-1].data.cpu()
        if flip:
            flip_input_var = torch.autograd.Variable(
                torch.from_numpy(fliplr(inputs.clone().numpy())).float(),
            )
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu(), meta_dir=data_dir)
            score_map += flip_output
            score_map /= 2

    if multi_scale:
        num_scales = len(scales)
        new_scales = []
        new_res = []
        new_score_map = []
        new_inp = []
        img_name = []
        confidence = []

        num_imgs = score_map.size(0) // num_scales
        for n in range(num_imgs):
            score_map_merged, res, conf = multi_scale_merge(score_map[num_scales * n: num_scales * (n + 1)].numpy(),
                                                            meta['scale'][num_scales * n: num_scales * (n + 1)])
            inp_merged, _, _ = multi_scale_merge(inputs[num_scales * n: num_scales * (n + 1)].numpy(),
                                                 meta['scale'][num_scales * n: num_scales * (n + 1)])
            new_score_map.append(score_map_merged)
            new_scales.append(meta['scale'][num_scales * (n + 1) - 1])
            new_res.append(res)
            new_inp.append(inp_merged)
            img_name.append(meta['img_name'][num_scales * n])
            confidence.append(conf)

        if len(new_score_map) > 1:
            score_map = torch.tensor(np.stack(new_score_map))  # stack back to 4-dim
            inputs = torch.tensor(np.stack(new_inp))
        else:
            score_map = torch.tensor(np.expand_dims(new_score_map[0], axis=0))
            inputs = torch.tensor(np.expand_dims(new_inp[0], axis=0))
    else:
        confidence = []
        for n in range(score_map.size(0)):
            confidence.append(np.amax(score_map[n].numpy(), axis=(1, 2)).tolist())

    if multi_scale:
        preds = final_preds(score_map, meta['center'], new_scales, new_res[0])
    else:
        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
        keypoints = preds[0].numpy()

    return np.matrix(keypoints).transpose()
