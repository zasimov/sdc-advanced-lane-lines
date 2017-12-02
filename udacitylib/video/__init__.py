"""udacitylib.video contains function to process video files

`convert` function processes video file using external function:

   convert(input_file_name, lambda x: x, output_file_name)

Note: convert function reads video file as a stream of BGR images

"""

from collections import namedtuple

import cv2


VideoProperties = namedtuple('VideoProperties', ['width', 'height', 'fps'])


def _vprops(vcap):
    if vcap.isOpened():
        # get vcap property
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = vcap.get(cv2.CAP_PROP_FPS)  # float

        return VideoProperties(width=width, height=height, fps=fps)

    raise ValueError('vcap is closed, sorry')


def convert(input_file, pipeline, output_file):
    """Converts input_file to output_file using pipeline"""
    input_ = cv2.VideoCapture(input_file)

    input_props = _vprops(input_)

    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_size = (int(input_props.width), int(input_props.height))
        out = cv2.VideoWriter(output_file, fourcc, int(input_props.fps), out_size)

        try:
            while input_.isOpened():
                ret, bgr_frame = input_.read()

                if not ret:
                    return

                out_frame = pipeline(bgr_frame)

                out.write(out_frame)
        finally:
            out.release()
    finally:
        input_.release()
