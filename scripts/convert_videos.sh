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
            -vcodec libxvid -b 18000k -acodec libmp3lame -ac 2 -ab 320k -deinterlace  -vf scale=160:-1 $output_video
     fi
done