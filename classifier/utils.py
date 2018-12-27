import collections
from functools import reduce
import operator
import tensorflow as tf

def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.
    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.
    Args:
        tensor: A tensor of any type.
    Returns:
        A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape

def pad_and_stack(tensors, stack_axis=-1):
    """Pad and stack a list of tensors."""
    shape_list = [combined_static_and_dynamic_shape(t) for t in tensors]
    shapes = tf.stack(shape_list, -1)
    max_shape = tf.reduce_max(shapes, axis=-1)
    output_tensor = []
    for shape, tensor in zip(shape_list, tensors):
        # Pad tensor
        paddings = tf.stack([tf.zeros_like(shape), max_shape - shape], axis=-1)
        output_tensor += [tf.pad(tensor, paddings)]
    return tf.stack(output_tensor, stack_axis)

def flatten_dict(d, parent_key='', sep='|'):
    """Flatten a nested dict."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def nest_dict(input_dict, sep='|'):
    """flatten_dict inverse function. Recursively nest dictionary according to sep.
    Args:
        input_dict: input possibly nested at different levels dictionary.
        sep: special characters used to join nested dictionaries
        into a single key.
    """
    def update(d, u):
        """Update current nested dict with another nested dict
        """
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    output_dict = {}
    for k, v in input_dict.items():
        if sep in k:
            nested_keys = k.split(sep)
            tree_dict = v
            for key in reversed(nested_keys):
                tree_dict = {key: tree_dict}
            update(output_dict, tree_dict)
        else:
            output_dict[k] = v
    return output_dict
