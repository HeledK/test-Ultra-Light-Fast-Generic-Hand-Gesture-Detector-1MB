import json

import numpy as np
import tensorflow as tf
import torch


def post_processing(reg_list, cls_list, num_classes, image_size, feature_map_wh_list, min_boxes,
                    center_variance, size_variance,
                    conf_threshold=0.6, nms_max_output_size=100, nms_iou_threshold=0.3, top_k=100):

    shapes = [512, 128, 48]
    input_shapes = [(15,20),(8,10),(4,5)]

    for r in reg_list:
        print(r.shape)

    reg_list = [tf.reshape(reg, shape=(int(shape/4),4)) for reg,shape, input_shape in zip(reg_list, shapes, input_shapes)]
    cls_list = [tf.reshape(cls, shape=(int(shape/4), num_classes)) for cls,shape, input_shape in zip(cls_list, shapes, input_shapes)]

    reg = tf.keras.layers.Concatenate(axis=0)(reg_list)
    cls = tf.keras.layers.Concatenate(axis=0)(cls_list)

    # post process
    cls = tf.keras.layers.Softmax(axis=-1)(cls)
    loc = decode_regression(reg, image_size, feature_map_wh_list, min_boxes,
                            center_variance, size_variance)

    result = tf.keras.layers.Concatenate(axis=-1)([cls, loc])

    return result


def decode_regression(reg, image_size, feature_map_w_h_list, min_boxes,
                      center_variance, size_variance):
    priors = []
    for feature_map_w_h, min_box in zip(feature_map_w_h_list, min_boxes):
        xy_grid = np.meshgrid(range(feature_map_w_h[0]), range(feature_map_w_h[1]))
        xy_grid = np.add(xy_grid, 0.5)
        xy_grid[0, :, :] /= feature_map_w_h[0]
        xy_grid[1, :, :] /= feature_map_w_h[1]
        xy_grid = np.stack(xy_grid, axis=-1)
        xy_grid = np.tile(xy_grid, [1, 1, len(min_box)])
        xy_grid = np.reshape(xy_grid, (-1, 2))

        wh_grid = np.array(min_box) / np.array(image_size)[:, np.newaxis]
        wh_grid = np.tile(np.transpose(wh_grid), [np.product(feature_map_w_h), 1])

        prior = np.concatenate((xy_grid, wh_grid), axis=-1)
        priors.append(prior)

    priors = np.concatenate(priors, axis=0)
    print(f'priors nums:{priors.shape[0]}')

    priors = tf.constant(priors, dtype=tf.float32, shape=priors.shape, name='priors')

    center_xy = reg[..., :2] * center_variance * priors[..., 2:] + priors[..., :2]
    center_wh = tf.exp(reg[..., 2:] * size_variance) * priors[..., 2:]

    # center to corner
    start_xy = center_xy - center_wh / 2
    end_xy = center_xy + center_wh / 2

    loc = tf.concat([start_xy, end_xy], axis=-1)
    loc = tf.clip_by_value(loc, clip_value_min=0.0, clip_value_max=1.0)

    return loc


def load_weight(model, torch_path, mapping_table_path):
    torch_weights = torch.load(torch_path, map_location=torch.device('cpu'))

    with open(mapping_table_path, 'r') as f:
        mapping_table = json.load(f)
        mapping_table = {layer['name']: layer['weight'] for layer in mapping_table}

    for layer in model.layers:
        if layer.name in mapping_table:
            print(f'Set layer: {layer.name}')
            layer_type = layer.name.split('_')[-1]

            torch_layer_names = mapping_table[layer.name]
            if layer_type == 'conv':
                weight = np.array(torch_weights[torch_layer_names[0]])
                weight = np.transpose(weight, [2, 3, 1, 0])
                layer.set_weights([weight])
            elif layer_type == 'dconv':
                weight = np.array(torch_weights[torch_layer_names[0]])
                weight = np.transpose(weight, [2, 3, 0, 1])
                layer.set_weights([weight])
            elif layer_type == 'bn':
                gamma = np.array(torch_weights[torch_layer_names[0]])
                beta = np.array(torch_weights[torch_layer_names[1]])
                running_mean = np.array(torch_weights[torch_layer_names[2]])
                running_var = np.array(torch_weights[torch_layer_names[3]])
                layer.set_weights([gamma, beta, running_mean, running_var])
            elif layer_type == 'convbias':
                weight = np.array(torch_weights[torch_layer_names[0]])
                bias = np.array(torch_weights[torch_layer_names[1]])
                weight = np.transpose(weight, [2, 3, 1, 0])
                layer.set_weights([weight, bias])
            elif layer_type == 'dconvbias':
                weight = np.array(torch_weights[torch_layer_names[0]])
                bias = np.array(torch_weights[torch_layer_names[1]])
                weight = np.transpose(weight, [2, 3, 0, 1])
                layer.set_weights([weight, bias])
            else:
                raise RuntimeError(f'Unknown Layer type \'{layer_type}\'.')
        else:
            print(f'Ignore layer: {layer.name}')
