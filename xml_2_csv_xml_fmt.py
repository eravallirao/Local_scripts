import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # i=0  # 	print(some)
        value = (root.find("path").text.split('\\')[-1],
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 root[1][0][0][0].text,
                 int(root[1][0][0][1][0].text),
                 int(root[1][0][0][1][1].text),
                 int(root[1][0][0][1][2].text),
                 int(root[1][0][0][1][3].text)
                 )
        # print(i)
        # if len(root[1][0][0])>1:
        xml_list.append(value)
    column_name = [ 'filename','width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    print(xml_df)
    return xml_df


def main():
    for directory in['train','test']:
        image_path = os.path.join(os.getcwd(), 'data/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/{}_labels.txt'.format(directory), index=None)
        print('Successfully converted xml to csv.')
main()
