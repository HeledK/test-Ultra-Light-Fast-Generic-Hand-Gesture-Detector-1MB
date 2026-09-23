import tensorflow as tf

from backend.op import conv_bn, conv_dw, separable_conv
from backend.utils import post_processing   #change this if want 320x240 to utils_test

conf_threshold = 0.6
nms_iou_threshold = 0.3
nms_max_output_size = 200
top_k = 100
center_variance = 0.1
size_variance = 0.2


image_size = [128, 128]  # MCU deployment input size
feature_map_wh_list = [[8, 8], [4, 4], [2, 2]]
min_boxes = [[32, 48], [64, 96], [128, 192, 256]]

#swap between the two if need to change input and need to change utils to utils_test at the top import

#image_size = [320, 240]   # (W, H) for 320×240 input
#feature_map_wh_list = [[20, 15], [10, 8], [5, 4]]   # (W, H) per scale
#min_boxes = [[32, 48], [64, 96], [128, 192, 256]]   # 2, 2, 3 anchors per scale


def create_slim_net_pruned(input_shape, widths, num_classes):
    """
    Pruned version of slim_320 for TF deployment.

    Args:
        input_shape: (H, W) tuple, e.g. (128, 128)
        widths: list of 13 ints — output channel widths for each backbone block.
                widths[7], widths[10], widths[12] MUST be 64, 128, 256 respectively
                (these are the FPN tap blocks that connect to detection heads).
        num_classes: number of detection classes (3 for background+palm+two)

    The PyTorch-side prunable indices are [0,1,2,3,4,5,6,8,9,11].
    Tap-fixed indices are 7, 10, 12.
    """
    assert len(widths) == 13, f'Expected 13 widths, got {len(widths)}'
    assert widths[7]  == 64,  f'widths[7] must be 64 (tap 1), got {widths[7]}'
    assert widths[10] == 128, f'widths[10] must be 128 (tap 2), got {widths[10]}'
    assert widths[12] == 256, f'widths[12] must be 256 (tap 3), got {widths[12]}'

    w = widths
    input_node = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 3))

    # Backbone — each block uses its pruned output width
    net = conv_bn(input_node, w[0], stride=2, prefix='basenet.0')
    net = conv_dw(net,        w[1], stride=1, prefix='basenet.1')
    net = conv_dw(net,        w[2], stride=2, prefix='basenet.2')
    net = conv_dw(net,        w[3], stride=1, prefix='basenet.3')
    net = conv_dw(net,        w[4], stride=2, prefix='basenet.4')
    net = conv_dw(net,        w[5], stride=1, prefix='basenet.5')
    net = conv_dw(net,        w[6], stride=1, prefix='basenet.6')
    header_0 = conv_dw(net,   w[7], stride=1, prefix='basenet.7')   # tap 1, FIXED 64
    net = conv_dw(header_0,   w[8], stride=2, prefix='basenet.8')
    net = conv_dw(net,        w[9], stride=1, prefix='basenet.9')
    header_1 = conv_dw(net,   w[10], stride=1, prefix='basenet.10') # tap 2, FIXED 128
    net = conv_dw(header_1,   w[11], stride=2, prefix='basenet.11')
    header_2 = conv_dw(net,   w[12], stride=1, prefix='basenet.12') # tap 3, FIXED 256

    # Extras block — input from header_2 (256ch), output FIXED at 256
    out = tf.keras.layers.Conv2D(64, 1, padding='SAME', name='extras_convbias')(header_2)
    out = tf.keras.layers.ReLU(name='extras_relu1')(out)
    out = separable_conv(out, 256, kernel_size=3, stride=2, padding=1,
                         prefix='extras_sep')
    header_3 = tf.keras.layers.ReLU(name='extras_relu2')(out)

    # Detection heads — input channels match the (fixed) tap outputs
    reg_0 = separable_conv(header_0, 3 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_0_sep')
    cls_0 = separable_conv(header_0, 3 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_0_sep')

    reg_1 = separable_conv(header_1, 2 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_1_sep')
    cls_1 = separable_conv(header_1, 2 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_1_sep')

    reg_2 = separable_conv(header_2, 2 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_2_sep')
    cls_2 = separable_conv(header_2, 2 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_2_sep')

    reg_3 = tf.keras.layers.Conv2D(3 * 4, kernel_size=3, padding='SAME',
                                   name='reg_3_convbias')(header_3)
    cls_3 = tf.keras.layers.Conv2D(3 * num_classes, kernel_size=3, padding='SAME',
                                   name='cls_3_convbias')(header_3)

    # Deployment uses only 3 detection scales (drops the finest header_0).
    # header_0's weights are still loaded for shape compatibility but unused at output.
    result = post_processing([reg_1, reg_2, reg_3],
                             [cls_1, cls_2, cls_3],
                             num_classes, image_size, feature_map_wh_list, min_boxes,
                             center_variance, size_variance)

    model = tf.keras.Model(inputs=[input_node], outputs=[result])
    model.summary()

    return model
