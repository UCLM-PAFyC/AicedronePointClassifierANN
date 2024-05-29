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


def classify(model_layers, ignored_scalar_fields, las: laspy.LasData, neighborhood_radius: float, model_path: str, output_path: str, save_features: bool, outlas: laspy.LasData, output_classification_field_name: str) -> None:
    """
    :param las: Input Las point cloud file
    :param neighborhood_radius: Neighborhood radius for computing the geometric features
    :param model_path: Path of the saved trained model
    :param output_path: Output LAS file path 
    """
    # Calculate geometric features
    print('Calculating geometric features...')
    computation.calculate_features(las, radius=neighborhood_radius, callback=show_percentage)

    # Classify
    print('Classifying point cloud...')
    classification = Classification(model_path, ignored_scalar_fields)
    if save_features:
        classification.classify(las, output_path, show_percentage)
    else:
        classification.classify_no_save_features(las, output_path, outlas, output_classification_field_name, show_percentage)


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
    parser.add_option("--input_point_cloud_file", dest="input_point_cloud_file", action="store", type="string",
                      help="Input point cloud file (las format)", default=None)
    parser.add_option("--scene_type", dest="scene_type", action="store", type="string",
                      help="Scene type to use", default=None)
    parser.add_option("--save_all_features", dest="save_all_features", action="store", type="string",
                      help="Save all features (0-Only classification in input field name in next argument, 1-all features used and prediction and probability)", default=None)
    parser.add_option("--output_classification_field_name", dest="output_classification_field_name", action="store", type="string",
                      help="Output classification field name, only for not save all features", default=None)
    parser.add_option("--output_point_cloud_file", dest="output_point_cloud_file", action="store", type="string",
                      help="Output point cloud file (las format)", default=None)
    parser.add_option("--neighborhood_radius", dest="neighborhood_radius", action="store", type="string",
                      help="Neighborhood_radius, in meters", default=None)
    parser.add_option("--input_model_path", dest="input_model_path", action="store", type="string",
                      help="Input model path", default=None)
    (options, args) = parser.parse_args()
    if not options.input_point_cloud_file:
        print("input_point_cloud_file")
        parser.print_help()
        return
    if not options.scene_type:
        print("scene_type")
        parser.print_help()
        return
    if not options.neighborhood_radius:
        print("neighborhood_radius")
        parser.print_help()
        return
    if not options.input_model_path:
        print("input_model_path")
        parser.print_help()
        return
    if not options.save_all_features:
        print("save_all_features")
        parser.print_help()
        return
    if not options.output_classification_field_name:
        print("output_classification_field_name")
        parser.print_help()
        return
    if not options.save_all_features:
        print("save_all_features")
        parser.print_help()
        return
    input_point_cloud_file = options.input_point_cloud_file
    if not exists(input_point_cloud_file):
        print("Error:\nNot exists input point cloud file:\n{}".format(input_point_cloud_file))
        return
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
    input_model_path = options.input_model_path
    if not os.path.exists(input_model_path):
        print("Not exists input model path:\n{}".format(input_model_path))
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
    input_las = laspy.read(input_point_cloud_file)
    str_save_all_features = options.save_all_features
    flag = True
    try:
        save_all_features = int(str_save_all_features)
    except ValueError:
        flag = False
    if not flag:
        print("Error:\nInvalid save_all_features: {}, must be 0/1".format(str_save_all_features))
        return
    if save_all_features < 0 or save_all_features > 1:
        print("Error:\nInvalid save_all_features: {}, must be 0/1".format(str_save_all_features))
        return
    save_features = True
    if save_all_features == 0:
        save_features = False
    output_las = None
    output_classification_field_name = None
    if not save_features:
        output_las = laspy.read(input_point_cloud_file)
        exists_ouput_class_field = False
        output_classification_field_name = options.output_classification_field_name
        for dimension in output_las.point_format:
            if dimension.name == output_classification_field_name:
                exists_ouput_class_field = True
                break
            else:
                if dimension.name == output_classification_field_name.lower():
                    output_classification_field_name = dimension.name
                    exists_ouput_class_field = True
                    break
                elif dimension.name == output_classification_field_name.upper():
                    output_classification_field_name = dimension.name
                    exists_ouput_class_field = True
                    break
        if not exists_ouput_class_field:
            output_las.add_extra_dim(laspy.ExtraBytesParams(name=output_classification_field_name,type=np.uchar,description="output class"))
    output_point_cloud_file = options.output_point_cloud_file
    if exists(output_point_cloud_file):
        os.remove(output_point_cloud_file)
        if exists(output_point_cloud_file):
            print("Error:\nError removing existing output file:\n{}".format(output_point_cloud_file))
            return
    model_layers = [
        InputLayer(neurons=64, activation='relu'),
        MultiHeadLayerDense(neurons=32, activation='relu', num_heads=4),
        HiddenLayer(neurons=64, activation='relu'),
        DropoutLayer(0.75),
        OutputLayer(activation='softmax')
    ]
    classify(model_layers, ignored_scalar_fields, input_las, neighborhood_radius=neighborhood_radius, model_path=input_model_path, output_path=output_point_cloud_file, save_features=save_features, outlas=output_las, output_classification_field_name=output_classification_field_name)
    elapsed = (time.time() - start)
    print("Elapsed time: {}".format(str(timedelta(seconds=elapsed))))
    
if __name__ == '__main__':
    main()
