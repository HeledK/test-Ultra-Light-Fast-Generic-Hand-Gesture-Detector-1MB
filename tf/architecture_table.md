| Type            | Output Shape          |   Count |   Total Params | First Layer          | Repeats   |
|:----------------|:----------------------|--------:|---------------:|:---------------------|:----------|
| InputLayer      | [(None, 128, 128, 3)] |       1 |              0 | input_1              |           |
| Conv2D          | (None, 64, 64, 16)    |       1 |            432 | basenet.0.0_conv     |           |
| DepthwiseConv2D | (None, 64, 64, 16)    |       1 |            144 | basenet.1.0_dconv    |           |
| Conv2D          | (None, 64, 64, 32)    |       1 |            512 | basenet.1.3_conv     |           |
| DepthwiseConv2D | (None, 32, 32, 32)    |       1 |            288 | basenet.2.0_dconv    |           |
| Conv2D          | (None, 32, 32, 32)    |       1 |           1024 | basenet.2.3_conv     |           |
| DepthwiseConv2D | (None, 32, 32, 32)    |       1 |            288 | basenet.3.0_dconv    |           |
| Conv2D          | (None, 32, 32, 32)    |       1 |           1024 | basenet.3.3_conv     |           |
| DepthwiseConv2D | (None, 16, 16, 32)    |       1 |            288 | basenet.4.0_dconv    |           |
| Conv2D          | (None, 16, 16, 64)    |       1 |           2048 | basenet.4.3_conv     |           |
| DepthwiseConv2D | (None, 16, 16, 64)    |       1 |            576 | basenet.5.0_dconv    |           |
| Conv2D          | (None, 16, 16, 64)    |       1 |           4096 | basenet.5.3_conv     |           |
| DepthwiseConv2D | (None, 16, 16, 64)    |       1 |            576 | basenet.6.0_dconv    |           |
| Conv2D          | (None, 16, 16, 64)    |       1 |           4096 | basenet.6.3_conv     |           |
| DepthwiseConv2D | (None, 16, 16, 64)    |       1 |            576 | basenet.7.0_dconv    |           |
| Conv2D          | (None, 16, 16, 64)    |       1 |           4096 | basenet.7.3_conv     |           |
| DepthwiseConv2D | (None, 8, 8, 64)      |       1 |            576 | basenet.8.0_dconv    |           |
| Conv2D          | (None, 8, 8, 128)     |       1 |           8192 | basenet.8.3_conv     |           |
| DepthwiseConv2D | (None, 8, 8, 128)     |       1 |           1152 | basenet.9.0_dconv    |           |
| Conv2D          | (None, 8, 8, 128)     |       1 |          16384 | basenet.9.3_conv     |           |
| DepthwiseConv2D | (None, 8, 8, 128)     |       1 |           1152 | basenet.10.0_dconv   |           |
| Conv2D          | (None, 8, 8, 128)     |       1 |          16384 | basenet.10.3_conv    |           |
| DepthwiseConv2D | (None, 4, 4, 128)     |       1 |           1152 | basenet.11.0_dconv   |           |
| Conv2D          | (None, 4, 4, 256)     |       1 |          32768 | basenet.11.3_conv    |           |
| DepthwiseConv2D | (None, 4, 4, 256)     |       1 |           2304 | basenet.12.0_dconv   |           |
| Conv2D          | (None, 4, 4, 256)     |       1 |          65536 | basenet.12.3_conv    |           |
| Conv2D          | (None, 4, 4, 64)      |       1 |          16448 | extras_convbias      |           |
| DepthwiseConv2D | (None, 2, 2, 64)      |       1 |            640 | extras_sep_dconvbias |           |
| Conv2D          | (None, 2, 2, 256)     |       1 |          16640 | extras_sep_convbias  |           |
| DepthwiseConv2D | (None, 8, 8, 128)     |       1 |           1280 | reg_1_sep_dconvbias  |           |
| DepthwiseConv2D | (None, 4, 4, 256)     |       1 |           2560 | reg_2_sep_dconvbias  |           |
| Conv2D          | (None, 8, 8, 8)       |       1 |           1032 | reg_1_sep_convbias   |           |
| Conv2D          | (None, 4, 4, 8)       |       1 |           2056 | reg_2_sep_convbias   |           |
| Conv2D          | (None, 2, 2, 12)      |       1 |          27660 | reg_3_convbias       |           |
| Concatenate     | (172, 4)              |       1 |              0 | concatenate          |           |
| DepthwiseConv2D | (None, 8, 8, 128)     |       1 |           1280 | cls_1_sep_dconvbias  |           |
| DepthwiseConv2D | (None, 4, 4, 256)     |       1 |           2560 | cls_2_sep_dconvbias  |           |
| Conv2D          | (None, 8, 8, 6)       |       1 |            774 | cls_1_sep_convbias   |           |
| Conv2D          | (None, 4, 4, 6)       |       1 |           1542 | cls_2_sep_convbias   |           |
| Conv2D          | (None, 2, 2, 9)       |       1 |          20745 | cls_3_convbias       |           |
| Concatenate     | (172, 3)              |       1 |              0 | concatenate_1        |           |
| Softmax         | (172, 3)              |       1 |              0 | softmax              |           |
| Concatenate     | (172, 7)              |       1 |              0 | concatenate_2        |           |