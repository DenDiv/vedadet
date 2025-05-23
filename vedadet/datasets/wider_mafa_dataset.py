import os.path as osp
import xml.etree.ElementTree as ET

import vedacore.fileio as fileio
from vedacore.misc import registry
from .xml_style import XMLDataset


@registry.register_module('dataset')
class WIDER_MAFA_Dataset(XMLDataset):
    """Reader for the WIDER+MAFA dataset in PASCAL VOC format.

    Source annotations could be found in FMLD_annotations.zip:
    https://github.com/borutb-fri/FMLD
    But I previously deleted WIDER annotations in this archive
    and renamed 'folder' attr
    """
    CLASSES = ('unmasked_face', 'masked_face', 'incorrectly_masked_face',)

    def __init__(self, **kwargs):
        super(WIDER_MAFA_Dataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from WIDERFace XML style annotation file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = fileio.list_from_file(ann_file)
        self.img_ids = img_ids
        for img_id in img_ids:
            filename = f'{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            folder = root.find('folder').text
            data_infos.append(
                dict(
                    id=img_id,
                    filename=osp.join(folder, filename),
                    width=width,
                    height=height))

        return data_infos
