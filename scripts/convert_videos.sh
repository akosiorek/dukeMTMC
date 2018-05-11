#!/usr/bin/env bash

dir=$1

for f in $(ls $dir)
do

    filename=$(basename -- "$f")
    extension="${filename##*.}"
    filename="${filename%.*}"


     if [[ "$extension" == "MTS" ]]
     then

        input_video=$dir/$f
        output_video=$dir/$filename.avi
        echo "Converting $f to $output_video"

        # convert MTS into AVI and downsample to 160 on the longer edge
        ffmpeg -i $input_video \
            -vcodec libx264 -b 18000k -vf scale=480:-1 $output_video
     fi
done
