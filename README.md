# Custom object detection with PyTorch and OID
The [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) is a dataset made by Google that provides more than 15 million boxes in 600 categories (more than 6,000 in the crowdsourced extended version). These annotations contain both detection boxes and segmentation. We may not need to download all classes to solve our detection task so it's a good idea to only get those classes that you really care about and avoid getting the whole dataset (+500Gb). There's a very cool repo that does exactly that [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit) but for now it's limited to V4 of the dataset (does not contain segmentation annotations). In order for us to train the model, we need to convert the data from OID to Pascal VOC (or MS-COCO).

## Requirements
- [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit)
- [SSD](https://github.com/lufficc/SSD)

## Download only selected classes from Open Images
Clone the `OIDv4_ToolKit` repo
```
!git clone https://github.com/EscVM/OIDv4_ToolKit
```

cd into `OIDv4_ToolKit` and run `python3 main.py downloader --classes {class} --type_csv all`
```
!cd OIDv4_ToolKit && python3 main.py downloader --classes Coin --type_csv all -y
```

We need to download extra files and place them in `~/OIDv4_ToolKit/OID/csv_folder`
```
!wget https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv
!wget https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
!wget https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv
```

## Convert OID to XML
Run `oid_to_pascal_voc_xml.py` ([source](https://gist.github.com/nilsfed/1dbf1cf397db50c90705daa6a81a8dec))from the `OIDv4_ToolKit` root directory.

The script will create directories called To_PASCAL_XML (similar to the Label directories) in the Dataset Subdirectories.
These directories contain the XML files.
You need to download the Images and generate ToolKit-Style-Labels via the OIDv4_ToolKit before using this script.

Run
```
!cd OIDv4_ToolKit && python oid_to_pascal_voc_xml.py
```

## Create VOC_ROOT folder structure and populate with data
```
...
path_to_OID = Path.cwd()/'object-detection/OIDv4_ToolKit/OID/'
image_sets = path_to_OID/'VOC_ROOT/VOC2012/ImageSets/Main'
jpeg_images = path_to_OID/'VOC_ROOT/VOC2012/JPEGImages'
annotations = path_to_OID/'VOC_ROOT/VOC2012/Annotations'

image_sets.mkdir(parents=True, exist_ok=True)
jpeg_images.mkdir(parents=True, exist_ok=True)
annotations.mkdir(parents=True, exist_ok=True)

for split in ['train','test','val']:
    list_images, list_xml = [],[]
    list_images = [x for x in (path_to_OID/f'Dataset/{split}/Coin').glob('*.jpg')]
    list_xml = [x for x in (path_to_OID/f'Dataset/{split}/Coin/To_PASCAL_XML').glob('*.xml')]
    
    with open(image_sets/f'{split}.txt', 'w') as f:
        for item in list_images:
            f.write("%s\n" % item.stem)
    
    for item in list_images:
        shutil.copyfile(item, jpeg_images/f'{item.name}')  
        
    for item in list_xml:
        shutil.copyfile(item, annotations/f'{item.name}')
        
with open(image_sets/'trainval.txt', 'wb') as wfd:
    for f in [image_sets/'train.txt',image_sets/'val.txt']:
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
...
```

## Train SSD model
The training with PASCAL VOC datasets need the class names to be in lower case in the XML so we are going to replace them in all files inside `VOC_ROOT/VOC2012/Annotations` with:
```
!find ./ -type f -exec sed -i 's/Coin/coin/' {} \;
```
Git clone the following SSD implementation in PyTorch:
```
git clone https://github.com/lufficc/SSD.git
cd SSD
#Required packages
pip install torch torchvision yacs tqdm opencv-python vizer
```
We need to edit the configuration file (.yaml): ie. `vgg_ssd300_voc0712.yaml` or `efficient_net_b3_ssd300_voc0712.yaml` and change `NUM_CLASSES:` from 21 to 2.
We now edit `./SSD/ssd/data/datasets/voc.py` and change:

```
class_names = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
```
to
```
class_names = ('__background__','coin')
```

We can start training with:

```
# for example, train SSD300:
!python train.py --config-file configs/vgg_ssd300_voc0712.yaml --use_tensorboard 0
```
