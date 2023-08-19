# Moire-Backdoor-Attack (MBA)
Official Pytorch implementation for our ACM MM 2023 paper: Moiré Backdoor Attack (MBA): A Novel Trigger for Pedestrian Detectors in the Physical World

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
Based on the above two datasets, two datasets, COCOPerson and OCHuman, were prepared in YOLO format, containing mask annotations.
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


<!-- # Citation
If you find the papers are useful for your research, please cite our papers as follows: -->

<!-- ```
<!-- @inproceedings{han2020neuromorphic,
  title={Moiré Backdoor Attack (MBA): A Novel Trigger for Pedestrian Detectors in the Physical World},
  author={Han, Jin and Zhou, Chu and Duan, Peiqi and Tang, Yehui and Xu, Chang and Xu, Chao and Huang, Tiejun and Shi, Boxin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
} -->
<!-- ``` --> 
