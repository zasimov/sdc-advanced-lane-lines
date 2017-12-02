#
# Script creates udacity camera model
#

$rows = 6
$cols = 9

# build camera model
python calibrate.py --chessboards camera_cal `
   --chessboards camera_cal `
   --rows $rows --cols $cols `
   --output camera.pickle
