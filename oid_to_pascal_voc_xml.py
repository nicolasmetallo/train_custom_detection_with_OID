"""
Location: OIDv4_ToolKit parent directory

Usage: 
    Start from OIDv4_ToolKit root directory.

    The script will create directories called To_PASCAL_XML (similar to the Label directories) in the Dataset Subdirectories.
    These directories contain the XML files.
    You need to download the Images and generate ToolKit-Style-Labels via the OIDv4_ToolKit before using this script.
    """

import os
from tqdm import tqdm
from sys import exit
import argparse
import cv2
from textwrap import dedent
from lxml import etree


XML_DIR = 'To_PASCAL_XML'


os.chdir(os.path.join("OID", "Dataset"))
DIRS = os.listdir(os.getcwd())

for DIR in DIRS:
    if os.path.isdir(DIR):
        os.chdir(DIR)

        print("Currently in Subdirectory:", DIR)

        CLASS_DIRS = os.listdir(os.getcwd())
        
        for CLASS_DIR in CLASS_DIRS:
            if os.path.isdir(CLASS_DIR):
                os.chdir(CLASS_DIR)

                print("\n" + "Creating PASCAL VOC XML Files for Class:", CLASS_DIR)
                # Create Directory for annotations if it does not exist yet
                if not os.path.exists(XML_DIR):
                    os.makedirs(XML_DIR)

                #Read Labels from OIDv4 ToolKit
                os.chdir("Label")

                #Create PASCAL XML
                for filename in tqdm(os.listdir(os.getcwd())):
                    if filename.endswith(".txt"):
                        filename_str = str.split(filename, ".")[0]


                        annotation = etree.Element("annotation")
                        
                        os.chdir("..")
                        folder = etree.Element("folder")
                        folder.text = os.path.basename(os.getcwd())
                        annotation.append(folder)

                        filename_xml = etree.Element("filename")
                        filename_xml.text = filename_str + ".jpg"
                        annotation.append(filename_xml)

                        path = etree.Element("path")
                        path.text = os.path.join(os.path.dirname(os.path.abspath(filename)), filename_str + ".jpg")
                        annotation.append(path)

                        source = etree.Element("source")
                        annotation.append(source)

                        database = etree.Element("database")
                        database.text = "Unknown"
                        source.append(database)

                        size = etree.Element("size")
                        annotation.append(size)

                        width = etree.Element("width")
                        height = etree.Element("height")
                        depth = etree.Element("depth")

                        img = cv2.imread(filename_xml.text)

                        width.text = str(img.shape[1])
                        height.text = str(img.shape[0])
                        depth.text = str(img.shape[2])

                        size.append(width)
                        size.append(height)
                        size.append(depth)

                        segmented = etree.Element("segmented")
                        segmented.text = "0"
                        annotation.append(segmented)

                        os.chdir("Label")
                        label_original = open(filename, 'r')

                        # Labels from OIDv4 Toolkit: name_of_class X_min Y_min X_max Y_max
                        for line in label_original:
                            line = line.strip()
                            l = line.split(' ')
                            class_name = l[0]
                            xmin_l = str(int(float(l[1])))
                            ymin_l = str(int(float(l[2])))
                            xmax_l = str(int(float(l[3])))
                            ymax_l = str(int(float(l[4])))
                            
                            obj = etree.Element("object")
                            annotation.append(obj)

                            name = etree.Element("name")
                            name.text = class_name
                            obj.append(name)

                            pose = etree.Element("pose")
                            pose.text = "Unspecified"
                            obj.append(pose)

                            truncated = etree.Element("truncated")
                            truncated.text = "0"
                            obj.append(truncated)

                            difficult = etree.Element("difficult")
                            difficult.text = "0"
                            obj.append(difficult)

                            bndbox = etree.Element("bndbox")
                            obj.append(bndbox)

                            xmin = etree.Element("xmin")
                            xmin.text = xmin_l
                            bndbox.append(xmin)

                            ymin = etree.Element("ymin")
                            ymin.text = ymin_l
                            bndbox.append(ymin)

                            xmax = etree.Element("xmax")
                            xmax.text = xmax_l
                            bndbox.append(xmax)

                            ymax = etree.Element("ymax")
                            ymax.text = ymax_l
                            bndbox.append(ymax)

                        os.chdir("..")

                        os.chdir(XML_DIR)

                        # write xml to file
                        s = etree.tostring(annotation, pretty_print=True)
                        with open(filename_str + ".xml", 'wb') as f:
                            f.write(s)
                            f.close()

                        os.chdir("..")
                        os.chdir("Label")

                os.chdir("..")
                os.chdir("..")   
                   
        os.chdir("..")
