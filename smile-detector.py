import cv2 as cv

cascade_file_smiley = cv.CascadeClassifier('smile.xml')
cascade_file_faces = cv.CascadeClassifier('face.xml')

image_path = cv.imread("faces/hero-bg.jpg")


def detect_smiling_face():
    gray_img = cv.cvtColor(image_path, cv.COLOR_BGR2GRAY)
    faces_detected = cascade_file_faces.detectMultiScale(gray_img, 1.1, 5)
    smiley_face = 0

    while len(faces_detected):
        for (x1, y1, w1, h1) in faces_detected:
            cv.rectangle(image_path, (x1, y1), (x1+w1, y1+h1), (0, 0, 245), 3)

            sub_frame = image_path[y1:y1+h1, x1:x1+w1]
            grayscale_img = cv.cvtColor(sub_frame, cv.COLOR_BGR2GRAY)
            smiley_face = cascade_file_smiley.detectMultiScale(grayscale_img, scaleFactor=1.5, minNeighbors=15)

            for (x2, y2, w2, h2) in smiley_face:
                cv.rectangle(sub_frame, (x2, y2), (x2 + w2, y2 + h2), (0, 210, 0), 3)
                cv.putText(sub_frame, "smile", (x2, y2), cv.FONT_ITALIC, 1.0, (255, 0, 0), 3)

        print(f'{len(smiley_face)} smiley face(s) detected.')
        print(f'{len(faces_detected)} face(s) detected.')

        # resize images...
        if image_path.shape[0] > 300 and image_path.shape[1] > 300:
            cv.imshow("Smiley", cv.resize(image_path, (700, 500)))
        else:
            cv.imshow("Smiley", image_path)

        if cv.waitKey(0) & 0xFF == ord('q'):
            break

    else:
        print('No face detected.')


detect_smiling_face()
