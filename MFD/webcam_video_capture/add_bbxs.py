import argparse

import cv2
import numpy as np
from vedacore.misc import Config, color_val
from vedacore.parallel import collate, scatter
from tools.infer import prepare
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Video detector')
    parser.add_argument('config', help='config file path')
    parser.add_argument('video_path', help='path to .avi video')
    args = parser.parse_args()
    return args


def plot_result(result, source_frame, class_names):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(source_frame, left_top, right_bottom, bbox_color, thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(source_frame, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    return source_frame


this_dir = os.path.dirname(os.path.abspath(__file__))
video_dir = os.path.join(this_dir, "videos")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    class_names = cfg.class_names
    vid = cv2.VideoCapture(args.video_path)

    if vid.isOpened() == False:
        print("Error opening video stream or file")

    engine, data_pipeline, device = prepare(cfg)

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    out = cv2.VideoWriter(os.path.join(video_dir, 'output_with_bbxs.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          20, (frame_width, frame_height))

    while vid.isOpened():

        ret, frame = vid.read()

        if ret == True:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = frame.astype(np.float32)
            data = dict(filename='0', ori_filename='0', img=frame, img_shape=frame.shape, ori_shape=frame.shape,
                img_fields=['img'])
            data = data_pipeline(data)
            data = collate([data], samples_per_gpu=1)

            if device != 'cpu':
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            else:
                # just get the actual data from DataContainer
                data['img_metas'] = data['img_metas'][0].data
                data['img'] = data['img'][0].data

            result = engine.infer(data['img'], data['img_metas'])[0]
            updated_frame = plot_result(result, frame, class_names)
            out.write(updated_frame)
        else:
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()
