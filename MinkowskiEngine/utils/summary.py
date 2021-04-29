import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

import MinkowskiEngine as ME
from MinkowskiSparseTensor import SparseTensor


def summary(model, summary_input):
    result, params_info = minkowski_summary_string(model, summary_input)
    print(result)
    return params_info


def pruned_weight_sparsity_string(module) -> float:
    r"""
    returns the sparsity ratio of weights.
    """
    for k in dir(module):
        if '_mask' in k:
            return (getattr(module, k.replace('_mask', '')) == 0).float().mean().item()
    else:
        return 0.0;


def size2list(size: torch.Size) -> list:
    return [i for i in size]

def get_hash_occupancy_ratio(minkowski_tensor):
    alg = minkowski_tensor.coordinate_manager.minkowski_algorithm
    if alg == ME.MinkowskiAlgorithm.SPEED_OPTIMIZED:
        return 25;
    else:
        return 50;

def minkowski_summary_string(model, summary_input):
    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            # for the weight pruned model, print the sparsity information
            summary[m_key]['sparsity_ratio'] = pruned_weight_sparsity_string(module)

            # save only the size of NNZ
            summary[m_key]["input_shape"] = input[0].shape
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [size2list(o.shape) for o in output]
            else:
                summary[m_key]["output_shape"] = size2list(output.shape)

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += module.weight.numel()
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "kernel") and hasattr(module.kernel, "size"):
                params += module.kernel.numel()
                summary[m_key]["trainable"] = module.kernel.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += module.bias.numel()
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(summary_input)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = (len(summary_input) * summary_input.shape[1]  # feature size
                        + len(summary_input) * (1 + summary_input.D) * (100 / get_hash_occupancy_ratio(summary_input)) # coordinate size
                       ) * 4. / (1024 ** 2.)
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)