dname=$1

case $dname in
    "st3d")
        gdown 1GTpjQrwvdRFex0st7rwYZ-7KoPbsPGRS
        unzip 03122_554516.zip
        rm -rf __MACOSX
        mkdir ~/data/st3d
        mv 03122_554516 ~/data/st3d
        rm 03122_554516.zip
        ;;
esac
