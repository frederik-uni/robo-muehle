#!/bin/bash

INPUT_DIR="game"
OUTPUT_GIF="game.gif"
FPS=4

i=0
for f in $(ls "$INPUT_DIR"/*.png | sort -V); do
    EXT="${f##*.}"
    mv "$f" "$(printf "$INPUT_DIR/%03d.$EXT" "$i")"
    ((i++))
done

ffmpeg -y -framerate $FPS -i "$INPUT_DIR/%03d.png" \
       -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
       "$OUTPUT_GIF"

echo "GIF created at $OUTPUT_GIF"
