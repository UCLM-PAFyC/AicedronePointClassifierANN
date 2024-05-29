"""
Train a neural network for point cloud classification
Alberto Morcillo Sanz - TIDOP
"""
from enum import Enum

import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras_multi_head import MultiHead

#import wandb
#from wandb.keras import WandbCallback

import laspy

import ann.scalarfields as scalarfields


class InputData:
    
    
    def __init__(self, las_training: laspy.LasData, las_test: laspy.LasData = None) -> None:
        """
        :param las_training: Training dataset
        :las_test: Test dataset
        """
        self.las_training = las_training
        self.las_test = las_test
        

class Layer:
    """
    Layer base class
    """
    
    class LayerType(Enum):
        INPUT = 1
        MULTI_HEAD_ATTENTION_DENSE = 2
        HIDDEN = 3
        OUTPUT = 4
        DROPOUT = 5,
        BATCH_NORMALIZATION = 6,
        CONV_1D = 7,
        GLOBAL_MAX_POOLING_1D = 8,
        CONCATENATION = 9
    
    
    def __init__(self, neurons: int, activation: str, layer_type: LayerType) -> None:
        """
        :param neurons: Number of neurons in the layer
        :param activation: Activation function
        :param layer_type:
        """
        self.neurons = neurons
        self.activation = activation
        self.layer_type = layer_type
        
        
class InputLayer(Layer):
    """
    Input Layer: Dense layer specifying the number of neurons and an activation function. 
    The real input layer is implicit (there is no need in specifying the input shape,
    __create_model does that for us)
    """
    
    def __init__(self, neurons: int, activation: str) -> None:
        """
        :param neurons: Number of neurons in the layer
        :param activation: Activation function
        """
        super().__init__(neurons, activation, Layer.LayerType.INPUT)
        

class MultiHeadLayerDense(Layer):
    """
    MultiHeadLayersDense: Multi-head attention layer, specifying the number of neurons, the 
    activation function of the dense layers and the number of heads
    
    If it is the first layer and there is no input layer. It adds an input layer automatically
    """
    
    def __init__(self, neurons: int, activation: str, num_heads: int = 8) -> None:
        super().__init__(neurons, activation, Layer.LayerType.MULTI_HEAD_ATTENTION_DENSE)
        self.num_heads = num_heads
        

class HiddenLayer(Layer):
    """
    HiddenLayer: Dense hidden layer, specifying the number of neurons and the activation function
    """
    
    def __init__(self, neurons: int, activation: str, l2: float = 0.0) -> None:
        """
        :param neurons: Number of neurons in the layer
        :param activation: Activation function
        """
        super().__init__(neurons, activation, Layer.LayerType.HIDDEN)
        self.l2 = l2


class OutputLayer(Layer):
    """
    OutputLayer: Last dense output, specifying the activation function (Softmax usually)
    """
    
    def __init__(self, activation: str) -> None:
        """
        :param activation: Activation function
        """
        super().__init__(0, activation, Layer.LayerType.OUTPUT)


class DropoutLayer(Layer):
    """
    DropoutLayer: DropoutLayer specifying the rate
    """
    
    def __init__(self, rate: float) -> None:
        """
        :param rate: randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
        """
        super().__init__(0, None, Layer.LayerType.DROPOUT)
        self.rate = rate


class Training:
    
    
    def __init__(self, input_data: InputData, label_title: str, ignored_scalar_fields: list[str], callback=None) -> None:
        """
        :param input_data: InputData structure with the training and test dataset
        :param label_title: Title of the label scalarfield
        :param percentageCallack: Captures the output
        """
        self.num_points_training: int = len(input_data.las_training.points)
        
        if input_data.las_test is None:
            self.__only_training_dataset(input_data.las_training, label_title, ignored_scalar_fields, callback)
        else:
            self.__training_and_test_dataset(input_data.las_training, input_data.las_test, label_title, ignored_scalar_fields, callback)
        
    
    def __only_training_dataset(self, las_training: laspy.LasData, label_title: str, ignored_scalar_fields: list[str], callback=None):
        """
        When the input data only has a training dataset (then it is splitted into train and test later)
        :param las_training: Training .LAS file
        :param label_title: Title of the label scalarfield
        :param percentageCallack: Captures the output
        """
        # Normalize data
        self.data: list[list[float]] = []
        scalarfields.normalize_scalar_fields(las_training, label_title, self.data, ignored_scalar_fields, callback)
        
        # Pandas dataframe
        titles: list[str] = [i for i in list(las_training.point_format.dimension_names) if i not in ignored_scalar_fields]
        df = pd.DataFrame(self.data, columns=titles)
        df[label_title] = df[label_title].astype(str)
        
        # Eliminar filas con valores NaN
        df_cleaned = df.dropna()

        # X: features Y: labels
        X = df_cleaned.iloc[:, :-1].values
        X = X.astype('float32')
        Y = df_cleaned.iloc[:, -1].values
        
        # Represent 1 as [1, 0, , ..., n], 2 as [0, 1, ..., n] ... where n is the number of classes
        Y = to_categorical(Y)
        
        # Separate train and test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.25)
        
              
    def __training_and_test_dataset(self, las_training: laspy.LasData, las_test: laspy.LasData, label_title: str, ignored_scalar_fields: list[str], callback=None) -> None:
        """
        When the input data has a train and test datasets
        :param las_training: Training .LAS file
        :param las_test: Test .LAS file
        :param label_title: Title of the label scalarfield
        :param callback: Captures the output
        """
        self.__load_training(las_training, label_title, ignored_scalar_fields, callback)
        self.__load_test(las_test, label_title, ignored_scalar_fields, callback)
        

    def __load_training(self, las_training: laspy.LasData, label_title: str, ignored_scalar_fields: list[str], callback=None):
        """
        :param las_training: Training .LAS file
        :param label_title: Title of the label scalarfield
        :param percentageCallack: Captures the output
        """
        training_data: list[list[float]] = []
        scalarfields.normalize_scalar_fields(las_training, label_title, training_data, ignored_scalar_fields, callback)

        titles: list[str] = [i for i in list(las_training.point_format.dimension_names) if i not in ignored_scalar_fields]
        df = pd.DataFrame(training_data, columns=titles)
        df[label_title] = df[label_title].astype(str)
        df_cleaned = df.dropna()
        
        self.x_train = df_cleaned.iloc[:, :-1].values
        self.x_train = self.x_train.astype('float32')
        
        self.y_train = df_cleaned.iloc[:, -1].values
        self.y_train = to_categorical(self.y_train)
        
        
    def __load_test(self, las_test: laspy.LasData, label_title: str, ignored_scalar_fields: list[str], callback=None):
        """
        :param las_test: Test .LAS file
        :param label_title: Title of the label scalarfield
        :param percentageCallack: Captures the output
        """
        test_data: list[list[float]] = []
        scalarfields.normalize_scalar_fields(las_test, label_title, test_data, ignored_scalar_fields, callback)
        
        titles: list[str] = [i for i in list(las_test.point_format.dimension_names) if i not in ignored_scalar_fields]
        df = pd.DataFrame(test_data, columns=titles)
        df[label_title] = df[label_title].astype(str)
        df_cleaned = df.dropna()
        
        self.x_test = df_cleaned.iloc[:, :-1].values
        self.x_test = self.x_test.astype('float32')
        
        self.y_test = df_cleaned.iloc[:, -1].values
        self.y_test = to_categorical(self.y_test)
            
        
    def __create_model(self, model_layers: list[any]) -> any:
        """
        Create neural network
        """
        
        has_input: bool = False
        model = tf.keras.Sequential()
        
        # Construct model from layers
        index: int = 0
        for model_layer in model_layers:
            
            # Input dense layer
            if model_layer.layer_type == Layer.LayerType.INPUT:
                model.add(layers.Dense(model_layer.neurons, input_dim=self.x_train.shape[1], activation=model_layer.activation))
                has_input = True
                
            # Multi head attention dense layers
            elif model_layer.layer_type == Layer.LayerType.MULTI_HEAD_ATTENTION_DENSE:
                
                # Add input layer if needed
                if not has_input:
                    model.add(layers.Input(shape=(self.x_train.shape[1],)))
                    has_input = True
            
                # MultiHead layer
                model.add(MultiHead(
                    layer=layers.Dense(model_layer.neurons, activation=model_layer.activation),
                    layer_num=model_layer.num_heads,
                    name='Multi-Head-Attention-Dense' + str(index),
                ))
                
                # Normalizationn layer
                model.add(layers.LayerNormalization())
                
                # Flatten layer
                model.add(layers.Flatten(name='Flatten' + str(index)))
                
            # Hidden dense layers
            elif model_layer.layer_type == Layer.LayerType.HIDDEN:
                
                hidden_layer = layers.Dense(model_layer.neurons, activation=model_layer.activation)
                    
                if model_layer.l2 != 0.0:
                    hidden_layer = layers.Dense(model_layer.neurons, activation=model_layer.activation, kernel_regularizer=regularizers.l2(model_layer.l2))
                
                model.add(hidden_layer)
                
            # Dropout layers
            elif model_layer.layer_type == Layer.LayerType.DROPOUT:
                model.add(layers.Dropout(model_layer.rate))
            
            # Output dense layers
            elif model_layer.layer_type == Layer.LayerType.OUTPUT:
                model.add(layers.Dense(self.y_train.shape[1], activation=model_layer.activation))
                
            index += 1
                
        model.build()
                
        # Compile model
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                'accuracy', 
                tf.keras.metrics.MeanIoU(num_classes=self.y_train.shape[1]), 
                tf.keras.metrics.Recall(),
                tf.keras.metrics.F1Score(threshold=0.5)
            ]
        )
        
        return model
    
    
    def __create_functional_model(self) -> any:
        
        # Input tensors
        input_data = layers.Input(shape=(self.x_train.shape[1],1))
        
        tensor_coords = layers.Reshape((1, 3))(input_data[:, :3])  # (batch_size, timesteps=1, x, y, z)
        tensor_features = layers.Reshape((1, self.x_train.shape[1]-3))(input_data[:, 3:])  # (batch_size, timesteps=1, r, g, b, f1, f2, ..., fn)
        
        # Convolution layers in features
        hidden_layer = layers.Conv1D(64, 1, activation='relu')(tensor_coords)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.Conv1D(128, 1, activation='relu')(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.Conv1D(512, 1, activation='relu')(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.Conv1D(1024, 1, activation='relu')(hidden_layer)
        
        # Global transformation in coords
        hidden_layer = layers.GlobalMaxPooling1D()(hidden_layer)
        
        # Concatenation of output_features and output_coords tensors
        output_coords = layers.Reshape((-1,))(hidden_layer)
        output_features = layers.Reshape((-1,))(tensor_features)
        concatenated_output = layers.Concatenate(axis=1)([output_features, output_coords])
        
        # Hidden layers
        hidden_layer = layers.Dense(128, 'relu')(concatenated_output)
        hidden_layer = layers.Dense(64, 'relu')(hidden_layer)
        hidden_layer = layers.Dropout(0.3)(hidden_layer)
        output_layer = layers.Dense(self.y_train.shape[1], 'softmax')(hidden_layer)
        
        # Model
        model = Model(inputs=input_data, outputs=output_layer)
        
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=self.y_train.shape[1])])
        
        return model
        
    
    def train(self, model_layers: list[any], epochs: int, batch_size: int, use_wandb: bool = False) -> any:
        """
        Train the neural network
        :param epochs: Number of epochs
        :param batch_size: The batch size
        :param use_wandb: Use Weights & Biases
        """
        
        '''
        wandb.init(
            project='PointClassifier-ANN', 
            config={
                "epochs": epochs,
            }
        )
        '''
        
        # Create neural network model
        self.model = self.__create_model(model_layers)
        #self.model = self.__create_functional_model()
        print(self.model.summary())
        
        # Train the model
        history = self.model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[]
            #callbacks=[WandbCallback()]
        )
        
        #wandb.finish()
        
        return history
    
    
    def saveModel(self, path: str) -> None:
        """
        Save a trained model in order to use it later
        :param path: Path where the model will be saved
        """
        self.model.save(path)