
**RoboMM**

In recent years, robotics has advanced significantly through the integration of larger models and large-scale datasets. However, challenges remain in applying these models to 3D spatial interactions and managing data collection costs. To address these issues, we propose the multimodal robotic manipulation model, **RoboMM**, along with the comprehensive dataset, **RoboData**.
**RoboMM** enhances 3D perception through camera parameters and occupancy supervision. Building on OpenFlamingo, it incorporates Modality-Isolation-Mask and multimodal decoder blocks, improving modality fusion and fine-grained perception. % , thus boosting performance in robotic manipulation tasks.
**RoboData** offers the complete evaluation system by integrating several well-known datasets, achieving the first fusion of multi-view images, camera parameters, depth maps, and actions, and the space alignment facilitates comprehensive learning from diverse robotic datasets.
Equipped with **RoboData** and the unified physical space, **RoboMM** is the first generalist policy that enables simultaneous evaluation across all tasks within multiple datasets, rather than focusing on limited selection of data or tasks.
Its design significantly enhances robotic manipulation performance, increasing the average sequence length on the CALVIN from 1.7 to 3.3 and ensuring cross-embodiment capabilities, achieving state-of-the-art results across multiple datasets. The code will be released following acceptance.


## Performance
![results](results.png)


## Training the model (using DDP):
```
bash tools/train.sh 8 --config ${config}

```

## Evaluating the model
```
bash tools/test.sh 8 ${ckpt}
```
