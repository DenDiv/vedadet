# prepare annotations in PASCAL VOC format for WIDER
python wider-face-pascal-voc-annotations/convert.py -ap wider-face-pascal-voc-annotations/wider_face_split/wider_face_train_bbx_gt.txt -tp data/WIDERFace/WIDER_train/Annotations/ -ip data/WIDERFace/WIDER_train --unmasked_mode 1 --skip_Surgeons 1 > /dev/null
python wider-face-pascal-voc-annotations/convert.py -ap wider-face-pascal-voc-annotations/wider_face_split/wider_face_val_bbx_gt.txt -tp data/WIDERFace/WIDER_val/Annotations/ -ip data/WIDERFace/WIDER_val --unmasked_mode 1 --skip_Surgeons 0 > /dev/null

ls -l data/WIDERFace/WIDER_train/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/WIDERFace/WIDER_train/train_WIDER.txt 
ls -l data/WIDERFace/WIDER_val/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/WIDERFace/WIDER_val/val_WIDER.txt 

python configs/trainval/tinaface/filter_widerface_val.py --gt_path eval_tools/ground_truth --ann_path data/WIDERFace/WIDER_val/Annotations/ > /dev/null

# prepare annotations in PASCAL VOC format for MAFA
unzip MAFA_anno.zip > /dev/null
mv MAFA_anno/train data/MAFA/MAFA_train/Annotations > /dev/null
mv MAFA_anno/test data/MAFA/MAFA_test/Annotations > /dev/null
rm -r MAFA_anno > /dev/null

ls -l data/MAFA/MAFA_train/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/MAFA/MAFA_train/train_MAFA.txt 
ls -l data/MAFA/MAFA_test/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/MAFA/MAFA_test/test_MAFA.txt 
echo "Finished"