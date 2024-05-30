# AicedronePointClassifierANN
Herramienta para la clasifación de nubes de puntos 3D utilizando redes neuronales artificiales
Esta versión ha sido desarrollada para su aplicación en el proyecto Aicedrone para la detección de raíles en vía ferroviaria

## Instalación
* Descargar el repositorio
* Crear un entorno de conda en miniconda ejecutando en la consola desde la ruta de descarga:

```
conda env create -f environment.yml
```

## Ejemplo de uso
* Descargar los datos de ejemplo de: https://drive.google.com/drive/folders/1msImEEzMFFp9f3xvUVCvM7kWPAbLoh3Y?usp=drive_link

* Entrenamiento, modificando las rutas y parámetros convenientemente:
```
* python train.py --point_cloud_file "C:\temp\example\input_training.las" --scene_type "railway" --field_name "Classification" --epochs 210 --batch_size 80000 --neighborhood_radius 0.030 --output_path "C:\temp\example\output_train" 
```
* Clasificación, modificando las rutas y parámetros convenientemente:
```
python .\src\console\predict.py --input_point_cloud_file "E:\python\Aicedrone_PC_ANN\example\input_classification.las" --scene_type "railway" --input_model_path "E:\python\Aicedrone_PC_ANN\example\output_train" --neighborhood_radius 0.030 --save_all_features 0 --output_classification_field_name "classification" --output_point_cloud_file "E:\python\Aicedrone_PC_ANN\example\output_classification.las" 
```

## **Alonso Garrido Limiñana**

Researcher, TIDOP

University of Salamanca - USAL, 
https://tidop.usal.es/miembros/alonso-garrido/

## **Alberto Morcillo Sanz**

Researcher, TIDOP

University of Salamanca - USAL, 
https://tidop.usal.es/miembros/alberto-morcillo-sanz/

## **David Hernández López**

Professor of Geomatics

University of Castilla-La Mancha - UCLM, 
david.hernandez@uclm.es

University of Salamanca - USAL, 
dhernand@usal.es

ORCID, [https://orcid.org/0000-0001-9874-5243](https://orcid.org/0000-0001-9874-5243)
