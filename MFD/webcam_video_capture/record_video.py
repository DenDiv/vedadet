import cv2
import numpy as np
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    video_dir = os.path.join(this_dir, "videos")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(os.path.join(video_dir, "output.avi"), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          30, (frame_width, frame_height))

    while (True):
        ret, frame = cap.read()

        if ret == True:

            # Write the frame into the file 'output.avi'
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()