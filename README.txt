To view the demo, please look at output.mp4

# To view the trajectory using the given code, please follow these instructions

cd cpp
#insert your dataset path in the indicated parts of actual_pose.cpp file
cmake -H. -Bbuild
cmake --build build -- -j3
cd build
./vo  
