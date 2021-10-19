import cv2
import dlib

# read the image
img = cv2.imread("222441155_3013665395589324_3963071017635881971_n.png")

#conver img to grayscalse: 3d --> 2D (B&W)
gray=  cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

#dlib: Load Face Recognition Detector
face_detector = dlib.get_frontal_face_detector()


#load the predictor: nhan dien đặc điểm mắt ,mũi, cằm ..

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#use detector to find face lamdmarks : nhung khuon mặt
faces =  face_detector(gray) # return những face nhận diện dc
# print(len(faces))

for face in faces:
    x1=face.left() # left Point
    y1=face.top()  # top Point
    x2= face.right() # right Point
    y2= face.bottom() # bottom Point

    #draw a rectangle
    cv2.rectangle(img= img, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness= 3)
    # break

    face_features= predictor(image = gray, box=face)

    # Loop through all 68 points
    for n in range(0,68):
        x=face_features.part(n).x
        y=face_features.part(n).y

        # Draw a circle 
        cv2.circle(img= img, center=(x,y), radius= 2 , color=(0,0,255),thickness=1)
# show the image
cv2.imshow(winname="Face Recognition App", mat = img)

# Note 1: "python.linting.pylintArgs": ["--generate-members"]

# wait for a key press to exit
cv2.waitKey(delay=0)

# Close All Windows
cv2.destroyAllWindows()