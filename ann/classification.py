"""
Classify a point cloud using a trained model
Alberto Morcillo Sanz - TIDOP
"""

import numpy as np

import tensorflow as tf

import laspy

import ann.scalarfields as scalarfields


class Classification:
    
    
    def __init__(self, model_path: str, ignored_scalar_fields: list[str]) -> None:
        """
        :param model_path: the model folder path
        """
        self.model_path = model_path
        self.ignored_scalar_fields = ignored_scalar_fields
        self.model = self.__load_model(model_path)
        print(self.model.summary())
        
    
    def __load_model(self, model_path: str) -> any:
        """
        Loads a previously trained model from a directory
        :param model_path: the model folder path
        """
        model = tf.keras.models.load_model(model_path)
        return model
    
    
    def classify(self, las: laspy.LasData, output_path: str, callback=None) -> None:
        """
        :param las: the las file of the point cloud we want to classify
        :param output_path: the output path
        :callback: callback function for tracking the percentage
        """
        # Normalizing data
        data: list[list[float]] = []
        scalarfields.normalize_scalar_fields(las, None, data, self.ignored_scalar_fields, callback)
        
        # Predict
        predictions = self.model.predict(data)
        
        # Create prediction scalar field in the .LAS file
        las.add_extra_dim(laspy.ExtraBytesParams(name='probability', type=np.float64))
        las.add_extra_dim(laspy.ExtraBytesParams(name='prediction', type=np.int64))
        
        # Get biggest probability class index in prediction
        num_points: int = len(las.points)
        previous_percentage: int = -1
        
        for i in range(0, num_points):
            
            prediction = predictions[i]
            class_index = np.argmax(prediction)
            
            # Save class index
            las['prediction'][i] = class_index
            las['probability'][i] = prediction[class_index]
            
            # Compute precentage
            if callback is not None:
                percentage = max(int(100 * float(i) / num_points) - 1, 0)
                if percentage != previous_percentage:
                    callback(percentage, self.classify)
                    previous_percentage = percentage
            
        # Save las
        las.write(output_path)
        
        # Compute percentage
        if callback is not None:
            callback(100, self.classify)
    
    
    def classify_no_save_features(self, las: laspy.LasData, output_path: str, outlas: laspy.LasData, output_class_field_name: str, callback=None) -> None:
        """
        :param las: the las file of the point cloud we want to classify
        :param output_path: the output path
        :callback: callback function for tracking the percentage
        """

        # Normalizing data
        data: list[list[float]] = []
        scalarfields.normalize_scalar_fields(las, None, data, self.ignored_scalar_fields, callback)
        
        # Predict
        predictions = self.model.predict(data)
        
        # # Create prediction scalar field in the .LAS file
        # las.add_extra_dim(laspy.ExtraBytesParams(name='probability', type=np.float64))
        # las.add_extra_dim(laspy.ExtraBytesParams(name='prediction', type=np.int64))
        
        # Get biggest probability class index in prediction
        num_points: int = len(las.points)
        previous_percentage: int = -1
        
        for i in range(0, num_points):
            
            prediction = predictions[i]
            class_index = np.argmax(prediction)
            
            # Save class index
            classification = round(class_index)
            if classification < 0:
                classification = 0
            elif classification > 255:
                classification = 255
            outlas[output_class_field_name][i] = round(class_index)
            # las['prediction'][i] = class_index
            # las['probability'][i] = prediction[class_index]
            
            # Compute precentage
            if callback is not None:
                percentage = max(int(100 * float(i) / num_points) - 1, 0)
                if percentage != previous_percentage:
                    callback(percentage, self.classify)
                    previous_percentage = percentage
            
        # Save las
        outlas.write(output_path)
        
        # Compute percentage
        if callback is not None:
            callback(100, self.classify)