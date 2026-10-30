# 下游检测可用性（YOLOv8x proxy）

| 条件 | n | R_any | R_aerial | IoU_best 均值 | IoU_best 中位 | IoU≥0.5 占比 |
|:--|--:|--:|--:|--:|--:|--:|
| _v9_prehalofix/out_D | 300 | 0.827 | 0.687 | 0.701 | 0.840 | 0.827 |
| out_D | 300 | 0.833 | 0.667 | 0.725 | 0.863 | 0.833 |
| out_D_halofix | 300 | 0.793 | 0.657 | 0.694 | 0.852 | 0.793 |

> ⚠️ COCO 预训练 YOLOv8x 无 drone 类，R_any / R_aerial 均为 proxy 口径，论文中须明确标注，不可直接表述为「无人机检测召回率」。
