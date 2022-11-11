import cv2

# Process the training data
# run edge detection on each frame. this may take a while. maybe keep a way to resume progress by letting the 
# read the ball image
ballimg = cv2.imread('example.png')
# 5x5 kernel size seems good
sobelx = cv2.Sobel(src=ballimg, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=ballimg, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

cv2.imwrite('sobelx.png', sobelx)
cv2.imwrite('sobely.png', sobely)

# user submit an index to resume at
# create a set of sets storing different animations and their frames + frame information by folder name and frame index
# alongside properties of that particular animation

# Gather the Training Data:
# Create an input set of two randomly selected frames from the same animation folder. store their indices.
# make sure any set is either before or after the first impact, or strictly contained between impacts. Impact frames
# are stored in a text file
# also access their location/size data from the planar image thing
# Their label should be the image associated with the label in the middle, with all its data as well
# create an array and memoize?

# Train the DCGAN