# PointClassifier-ANN
Point cloud classification with deep learning

![](img/img.png)

## Features
* Compute geometric features
* Sequential model
* Functional model
* Multi-Head Attention layers
* Convolution layers
* Global Max Pooling layers
* Dense layers
* Normalization layers
* Batch normalization layers
* Dropouts

## TODO
* Improve model
* Avoid more overfitting
* Remove noise using Graph Cut

# Dependencies

```
conda env create -f environment.yml
```

python .\src\console\train.py --point_cloud_file "E:\python\Aicedrone_PC_ANN\example\input_training.las" --scene_type "railway" --field_name "Classification" --epochs 210 --batch_size 80000 --neighborhood_radius 0.030 --output_path "E:\python\Aicedrone_PC_ANN\example\output_train" 

python .\src\console\predict.py --input_point_cloud_file "E:\python\Aicedrone_PC_ANN\example\input_classification.las" --scene_type "railway" --input_model_path "E:\python\Aicedrone_PC_ANN\example\output_train" --neighborhood_radius 0.030 --save_all_features 0 --output_classification_field_name "classification" --output_point_cloud_file "E:\python\Aicedrone_PC_ANN\example\output_classification.las" 
