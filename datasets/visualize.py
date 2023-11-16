from voc import VOCSegmentation
from voc import transform_img_mask
import torch.utils.data as data
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def parse_xml(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Walk through the XML tree
    xml_data = {
        'filename': root.find('filename').text,
        'size': {
            'width': int(root.find('size/width').text),
            'height': int(root.find('size/height').text),
            'depth': int(root.find('size/depth').text)
        },
        'objects': []
    }

    for obj_elem in root.findall('object'):

        difficult = None
        if(obj_elem.find('difficult')):
            difficult = int(obj_elem.find('difficult').text)
        truncated = None
        if(obj_elem.find('truncated')):
            truncated = int(obj_elem.find('truncated').text)

        obj = {
            'name': obj_elem.find('name').text,
            'pose': obj_elem.find('pose').text,
            'truncated': truncated,
            'difficult': difficult,
            'bndbox': {
                'xmin': float(obj_elem.find('bndbox/xmin').text),
                'ymin': float(obj_elem.find('bndbox/ymin').text),
                'xmax': float(obj_elem.find('bndbox/xmax').text),
                'ymax': float(obj_elem.find('bndbox/ymax').text),
            }
        }
        xml_data['objects'].append(obj)

    return xml_data


def plot_images_with_boxes(image_data_list, num_columns=5, num_rows=4):
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 8))
    fig.subplots_adjust(hspace=0.5)

    for i, xml_data in enumerate(image_data_list):
        # Load image
        image_path = "data/VOCdevkit/VOC2012/JPEGImages/" + xml_data['filename']
        img = Image.open(image_path)

        # Plot image
        ax = axs[i // num_columns, i % num_columns]
        ax.imshow(img)
        ax.axis('off')

        # Plot bounding boxes
        for obj in xml_data['objects']:
            xmin, ymin, xmax, ymax = obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'], obj['bndbox']['ymax']
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Display object name
            ax.text(xmin, ymin - 5, obj['name'], color='r', fontsize=8, fontweight='bold')

    plt.show()


if __name__ == "__main__":
    folder_path = "data/VOCdevkit/VOC2012/Annotations/"

    all_xml_data = []
    num_of_images = 20
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            xml_data = parse_xml(file_path)
            all_xml_data.append(xml_data)
            num_of_images -= 1

        if num_of_images <= 0:
            break

    plot_images_with_boxes(all_xml_data)








