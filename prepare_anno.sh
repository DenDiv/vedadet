# prepare annotations in PASCAL VOC format for WIDER
python wider-face-pascal-voc-annotations/convert.py -ap wider-face-pascal-voc-annotations/wider_face_split/wider_face_train_bbx_gt.txt -tp data/WIDERFace/WIDER_train/Annotations/ -ip data/WIDERFace/WIDER_train --unmasked_mode 1 --skip_Surgeons 1
python wider-face-pascal-voc-annotations/convert.py -ap wider-face-pascal-voc-annotations/wider_face_split/wider_face_val_bbx_gt.txt -tp data/WIDERFace/WIDER_val/Annotations/ -ip data/WIDERFace/WIDER_val --unmasked_mode 1 --skip_Surgeons 0

ls -l data/WIDERFace/WIDER_train/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/WIDERFace/WIDER_train/train_WIDER.txt
ls -l data/WIDERFace/WIDER_val/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/WIDERFace/WIDER_val/val_WIDER.txt

python configs/trainval/tinaface/filter_widerface_val.py --gt_path eval_tools/ground_truth --ann_path data/WIDERFace/WIDER_val/Annotations/

# prepare annotations in PASCAL VOC format for MAFA
unzip MAFA_anno.zip
mv MAFA_anno/train data/MAFA/MAFA_train/Annotations
mv MAFA_anno/test data/MAFA/MAFA_test/Annotations
rm -r MAFA_anno

ls -l data/MAFA/MAFA_train/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/MAFA/MAFA_train/train_MAFA.txt
ls -l data/MAFA/MAFA_test/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/MAFA/MAFA_test/test_MAFA.txt