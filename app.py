import streamlit as st
import cv2
import numpy as np
import utils


# Define Streamlit app
def main():
    st.title("Mark Sheet Scanner")
    st.write("Upload a mark sheet image to extract marks and grade.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a mark sheet image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Display uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Process the image
        widthImg, heightImg = 700, 700
        img = cv2.resize(img, (widthImg, heightImg))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 10, 50)

        # Find contours
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rectCon = utils.rectContour(contours)
        if len(rectCon) < 2:
            st.error("Unable to find the required contours.")
            return

        biggestContour = utils.getCornerPoints(rectCon[0])
        gradePoints = utils.getCornerPoints(rectCon[1])

        if biggestContour.size == 0 or gradePoints.size == 0:
            st.error("Contours for marks and grades could not be detected.")
            return

        biggestContour = utils.reorder(biggestContour)
        gradePoints = utils.reorder(gradePoints)

        # Warp the perspective
        pt1 = np.float32(biggestContour)
        pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # Thresholding
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

        # Split boxes
        questions, choices = 5, 5
        ans = [1, 2, 0, 1, 4]
        boxes = utils.splitBoxes(imgThresh)

        # Calculate pixel values
        myPixelVal = np.zeros((questions, choices))
        countC, countR = 0, 0
        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC += 1
            if countC == choices:
                countR += 1
                countC = 0

        # Find marked answers
        myIndex = []
        for x in range(questions):
            arr = myPixelVal[x]
            myIndexVal = np.where(arr == np.amax(arr))
            myIndex.append(myIndexVal[0][0])

        # Grade calculation
        grading = []
        for x in range(questions):
            if ans[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)
        score = (sum(grading) / questions) * 100

        # Display results
        st.subheader("Results")
        st.write(f"Extracted Answers: {myIndex}")
        st.write(f"Correct Answers: {ans}")
        st.write(f"Grade: {score:.2f}%")

        # Display final marked image
        imgResult = imgWarpColored.copy()
        utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
        st.image(imgResult, caption="Processed Marked Sheet", use_column_width=True)


# Run the app
if __name__ == "__main__":
    main()
