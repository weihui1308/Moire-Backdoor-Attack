# Moire-Backdoor-Attack (MBA)
Official Pytorch implementation for our ACM MM 2023 paper: [Moiré Backdoor Attack (MBA): A Novel Trigger for Pedestrian Detectors in the Physical World](https://dl.acm.org/doi/10.1145/3581783.3611910)

# Samples
![Figure](https://github.com/weihui1308/Moire-Backdoor-Attack/blob/main/assets/display.png?raw=true)

# Preparation
- Python
- Pytorch
- YOLOv5
- MMDetection

# Usage
Please download the COCO2017 dataset in this [link](https://cocodataset.org/#download) and the OCHuman in this [link](https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman).
- Step 0:  
Creating COCOPerson and OCHuman datasets in YOLO format, complete with mask labels, based on the aforementioned two datasets.
- Step 1:  
Generating poisoned data samples.  
    ```shell
    python moire2img.py --source imageDir
    ```
- Step 2:  
Generating poisoned dataset and put in the ```dataset``` folder.
    ```shell
    python move2dataset.py
    ```

- Step 3:  
Train on the poisoned dataset using the corresponding scripts of the two libraries [YOLOv5](https://github.com/ultralytics/yolov5) and [MMDetection](https://github.com/open-mmlab/mmdetection).


# Citation
If you find the papers are useful for your research, please cite our paper as follows:

```
@inproceedings{wei2023moire,
  title={Moir{\'e} Backdoor Attack (MBA): A Novel Trigger for Pedestrian Detectors in the Physical World},
  author={Wei, Hui and Yu, Hanxun and Zhang, Kewei and Wang, Zhixiang and Zhu, Jianke and Wang, Zheng},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={8828--8838},
  year={2023}
}
``` 

# Acknowledgements
We would like to acknowledge the YOLOv5 open-source library (https://github.com/ultralytics/yolov5) and MMDetection open-source library (https://github.com/open-mmlab/mmdetection).
