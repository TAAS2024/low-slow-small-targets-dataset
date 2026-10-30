# 下游检测可用性（YOLOv8x proxy）

| 条件 | n | R_any | R_aerial | IoU_best 均值 | IoU_best 中位 | IoU≥0.5 占比 |
|:--|--:|--:|--:|--:|--:|--:|
| _v9_prehalofix/out_D | 300 | 0.650 | 0.557 | 0.552 | 0.789 | 0.650 |
| out_D | 300 | 0.617 | 0.507 | 0.543 | 0.805 | 0.617 |
| out_D_halofix | 300 | 0.620 | 0.520 | 0.542 | 0.792 | 0.620 |

> ⚠️ COCO 预训练 YOLOv8x 无 drone 类，R_any / R_aerial 均为 proxy 口径，论文中须明确标注，不可直接表述为「无人机检测召回率」。
