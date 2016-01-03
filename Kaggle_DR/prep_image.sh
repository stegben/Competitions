#!/bin/sh -e 

# To run, this assumes that you are in the directory with the images
# unpacked into train/ and test/.  To run, it works best to use GNU
# parallel as
#
#   ls train/*.jpeg test/*.jpeg | parallel ./prep_image.sh
#   ls sample/*.jpeg | parallel ./prep_image.sh
#
# Otherwise, it also works to do a bash for loop, but this is slower.
#
#   for f in `ls train/*.jpeg test/*.jpeg`; do ./prep_image.sh $f; done
#   for f in `ls sample/*.jpeg`; do ./prep_image.sh $f; done
#

size=512x512

out_dir=proc_gamma/green
mkdir -p $out_dir/train
mkdir -p $out_dir/test
mkdir -p $out_dir/sample
out=$out_dir/$1
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
convert -fuzz 8%  -trim +repage -transparent black -resize $size -gravity center -background white -extent $size -equalize -separate -delete 0,2 $1 $out
# convert -trim +repage -resize $size -gravity center -background white -extent $size -equalize $1 $out

out_dir=proc_small_gamma/normal
mkdir -p $out_dir/train
mkdir -p $out_dir/test
mkdir -p $out_dir/sample
out=$out_dir/$1
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
convert -fuzz 8% -auto-gamma -trim +repage -transparent black -resize $size -gravity center -background white -extent $size -equalize $1 $out


size=256x256

out_dir=proc_small_gamma/normal
mkdir -p $out_dir/train
mkdir -p $out_dir/test
mkdir -p $out_dir/sample
out=$out_dir/$1
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
convert -fuzz 8% -auto-gamma -trim +repage -transparent black -resize $size -gravity center -background white -extent $size -equalize $1 $out

out_dir=proc_small_gamma/green
mkdir -p $out_dir/train
mkdir -p $out_dir/test
mkdir -p $out_dir/sample
out=$out_dir/$1
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
convert -fuzz 8% -auto-gamma -trim +repage -transparent black -resize $size -gravity center -background white -extent $size -equalize -separate -delete 0,2 $1 $out

