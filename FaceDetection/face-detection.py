import cv2

image = cv2.imread('pessoas.jpg')

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#scaleFactor = scale image appplied
#minNeighbors = min quantity of rectangles neighbors
#minZise = min size of the image to be detect. Default size = 30 x 30.
#maxZise = max size of the image to be detect
detections = classifier.detectMultiScale(grayImage,
                                         scaleFactor = 1.09,
                                         minNeighbors = 5,
                                         minSize = (30,30),
                                         maxSize = (40,40))

print(detections)
print(len(detections))

#create bounding box
#left = x, top = y
for(left, top, width, height) in detections:
    cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)

cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()