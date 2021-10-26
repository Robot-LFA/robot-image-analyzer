from backend import *
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('input_folder', help='input folder containing images')
parser.add_argument('parameters', help='parameters for image analysis')
args = parser.parse_args()

print(args)

para = json.loads(args.parameters)


def method(image):
    return process_image(image, **para)


process_folder(args.input_folder, method, 'image')

