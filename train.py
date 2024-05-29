"""
Point cloud classification
Alberto Morcillo Sanz, amorcillosanz@gmail.com
David Hernandez Lopez, david.hernandez@uclm.es
"""

import optparse
import sys
import os
current_path = os.path.abspath(os.path.join(__file__ ,"../.."))
sys.path.insert(0, current_path)
from os.path import exists
from math import floor, ceil, sqrt, isnan, modf, trunc, sin, cos
import re
import laspy
import matplotlib.pyplot as plt
import numpy as np
import time
import shutil
from datetime import timedelta
import geometric.computation as computation
from ann.training import Training
from ann.training import InputData
from ann.training import InputLayer
from ann.training import HiddenLayer
from ann.training import OutputLayer
from ann.training import DropoutLayer
from ann.training import MultiHeadLayerDense
from ann.classification import Classification


class OptionParser(optparse.OptionParser):
    def check_required(self, opt):
        option = self.get_option(opt)
        if getattr(self.values, option.dest) is None:
            self.error("%s option not supplied" % option)


def is_number(n):
    is_number = True
    try:
        num = float(n)
        is_number = num == num 
    except ValueError:
        is_number = False
    return is_number


def show_percentage(percentage: int, function=None) -> None:
    """
    show_percentage is the callback sent to computation.calculateFeatrues in order
    to track the progress of the computation
    :param percentage: percentage to show
    """
    if function is None:
        print('Progress:', str(percentage) + '%')
    else:
        print(function.__name__ + ': ' + str(percentage) + '%')


def train(model_layers, ignored_scalar_fields, las: laspy.LasData, neighborhood_radius: float, epochs: int, batch_size: int, label_title: str, output_path: str):
    """
    :param las: Las training file
    :param neighborhood_radius: Neighborhood radius for computing the geometric features
    :param epochs: Number of epochs for training
    :param batch_size: Batch size
    :param label_title: Label scalar field name for spliting the features from the labels
    :parm output_path: Path where the model will be saved
    """
    # Calculate geometric features
    print('Calculating geometric features...')
    computation.calculate_features(las, radius=neighborhood_radius, callback=show_percentage)

    #  Training
    print('Training neural network')
    training: Training = Training(InputData(las), label_title, ignored_scalar_fields, show_percentage)
    history = training.train(model_layers, epochs, batch_size, use_wandb=True)

    # Plot training
    print(history.history.keys())
    plt.plot(history.history["loss"])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['loss', 'accuracy', 'val_loss', 'val_accuracy'], loc='upper left')

    # Save model
    training.saveModel(output_path)


def main():
    # ==================
    # parse command line
    # ==================
    ignored_scalar_fields_by_scene_type = {}
    ignored_scalar_fields_by_scene_type['railway'] = ['X', 'Y', 'return_number', 'number_of_returns', 'scan_direction_flag', 'edge_of_flight_line', 
                          'classification', 'synthetic', 'key_point', 'withheld', 'scan_angle_rank', 'user_data', 'point_source_id',
                          'Distance_to_plane']
    start = time.time()
    usage = "usage: %prog [options] "
    parser = OptionParser(usage=usage)
    parser.add_option("--point_cloud_file", dest="point_cloud_file", action="store", type="string",
                      help="Point cloud file (las format)", default=None)
    parser.add_option("--field_name", dest="field_name", action="store", type="string",
                      help="Field name to use", default=None)
    parser.add_option("--scene_type", dest="scene_type", action="store", type="string",
                      help="Scene type to use", default=None)
    parser.add_option("--epochs", dest="epochs", action="store", type="string",
                      help="Integer for number of epochs",
                      default=None)
    parser.add_option("--batch_size", dest="batch_size", action="store", type="string",
                      help="Integer for batch size",
                      default=None)
    parser.add_option("--neighborhood_radius", dest="neighborhood_radius", action="store", type="string",
                      help="Neighborhood_radius, in meters", default=None)
    parser.add_option("--output_path", dest="output_path", action="store", type="string",
                      help="Path for output", default=None)
    (options, args) = parser.parse_args()
    if not options.point_cloud_file:
        parser.print_help()
        return
    if not options.field_name:
        parser.print_help()
        return
    if not options.scene_type:
        parser.print_help()
        return
    if not options.epochs:
        parser.print_help()
        return
    if not options.batch_size:
        parser.print_help()
        return
    if not options.neighborhood_radius:
        parser.print_help()
        return
    if not options.output_path:
        parser.print_help()
        return
    point_cloud_file = options.point_cloud_file
    if not exists(point_cloud_file):
        print("Error:\nNot exists point cloud file:\n{}".format(point_cloud_file))
        return
    field_name = options.field_name
    str_epochs = options.epochs
    flag = True
    try:
        epochs = int(str_epochs)
    except ValueError:
        flag = False
    if not flag:
        print("Error:\nInvalid epochs: {}".format(str_epochs))
        return
    str_batch_size = options.batch_size
    flag = True
    try:
        batch_size = int(str_batch_size)
    except ValueError:
        flag = False
    if not flag:
        print("Error:\nInvalid batch_size: {}".format(str_batch_size))
        return
    str_neighborhood_radius = options.neighborhood_radius
    flag = True
    try:
        neighborhood_radius = float(str_neighborhood_radius)
    except ValueError:
        flag = False
    if not flag:
        print("Error:\nInvalid neighborhood radius: {}".format(str_neighborhood_radius))
        return
    input_las = laspy.read(point_cloud_file)
    exists_field = False
    for dimension in input_las.point_format:
        if dimension.name == field_name:
            exists_field = True
            break
        else:
            if dimension.name == field_name.lower():
                field_name = dimension.name
                exists_field = True
                break
            elif dimension.name == field_name.upper():
                field_name = dimension.name
                exists_field = True
                break
    if not exists_field:
        print("Error:\nNot exists field: {} in  file:\n{}".format(field_name,point_cloud_file))
        return
    new_input_class_field_name = "input_class"
    exists_new_input_class_field = False
    for dimension in input_las.point_format:
        if dimension.name == new_input_class_field_name:
            exists_new_input_class_field = True
            break
        else:
            if dimension.name == new_input_class_field_name.lower():
                new_input_class_field_name = dimension.name
                exists_new_input_class_field = True
                break
            elif dimension.name == new_input_class_field_name.upper():
                new_input_class_field_name = dimension.name
                exists_new_input_class_field = True
                break
    scene_type = options.scene_type
    ignored_scalar_fields = None
    scene_types = ignored_scalar_fields_by_scene_type.keys()
    for st in scene_types:
        if scene_type.lower() == st.lower():
            ignored_scalar_fields = ignored_scalar_fields_by_scene_type[st]
            break;
    if not ignored_scalar_fields:
        print("Invalid scene type: {}\nOptions: {}".format(scene_type,scene_types))
        return
    output_path = options.output_path
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        if os.path.exists(output_path):
            print("Error removing the existing output path:\n{}".format(output_path))
            return
    os.makedirs(output_path)
    if not os.path.exists(output_path):
        print("Error creating the output path:\n{}".format(output_path))
        return
    if not exists_new_input_class_field:
        input_las.add_extra_dim(laspy.ExtraBytesParams(name=new_input_class_field_name,type=np.uint,description="new input class"))
    class_values = input_las.__getattr__(field_name)
    input_las.__setattr__(new_input_class_field_name,class_values)
    model_layers = [
        InputLayer(neurons=64, activation='relu'),
        MultiHeadLayerDense(neurons=32, activation='relu', num_heads=4),
        HiddenLayer(neurons=64, activation='relu'),
        DropoutLayer(0.75),
        OutputLayer(activation='softmax')
    ]
    train(model_layers, ignored_scalar_fields, input_las, neighborhood_radius=neighborhood_radius, epochs=epochs, batch_size=batch_size, label_title=new_input_class_field_name, output_path=output_path)
    elapsed = (time.time() - start)
    print("Elapsed time: {}".format(str(timedelta(seconds=elapsed))))

if __name__ == '__main__':
    main()
