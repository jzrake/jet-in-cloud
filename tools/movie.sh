#!/bin/bash

ffmpeg -r 12 -f image2 -s 1434x826 -i frames/frame.%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p out.mp4
