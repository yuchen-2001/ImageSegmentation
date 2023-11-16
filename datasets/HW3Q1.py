import xml.etree.ElementTree as ET
import os

def parse_xml(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    all_categoreis = []

    for obj_elem in root.findall('object'):
        name = obj_elem.find('name').text
        all_categoreis.append(name)
        return all_categoreis

    return all_categoreis


def process_folder(folder_path):
    all_xml_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            try:
                xml_data = parse_xml(file_path)
                all_xml_data = all_xml_data + xml_data
            except ET.ParseError as e:
                print(f"Error parsing XML file {filename}: {e}")

    return all_xml_data


if __name__ == "__main__":
    # Provide the path to your XML file
    folder_path = "data/VOCdevkit/VOC2012/Annotations/"

    categoreis_list = process_folder(folder_path)
    all_categories = set()
    for element in categoreis_list:
        all_categories.add(element)
    print(all_categories)