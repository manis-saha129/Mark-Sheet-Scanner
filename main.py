import cv2
import numpy as np
import utils

# Variables
path = "test/1.jpg"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [1, 2, 0, 1, 4]

img = cv2.imread(path)

# Preprocessing
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# Finding all contours
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

# Find rectangles
rectCon = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
print(biggestContour.shape)
gradePoints = utils.getCornerPoints(rectCon[1])
print(biggestContour)
print(len(biggestContour))

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

    biggestContour = utils.reorder(biggestContour)
    gradePoints = utils.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
    cv2.imshow("Grade", imgGradeDisplay)

    # Apply threshold
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = utils.splitBoxes(imgThresh)
    # cv2.imshow("Test", boxes[2])
    print(cv2.countNonZero(boxes[1]), cv2.countNonZero(boxes[2]))

    # Getting the non-zero pixel values of each box
    myPixelVal = np.zeros((questions, choices))
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if countC == choices:
            countR += 1
            countC = 0
    print(myPixelVal)

    # Finding the index values of the markings
    myIndex = []
    for x in range(0, questions):
        arr = myPixelVal[x]
        print("Array", arr)
        myIndexVal = np.where(arr == np.amax(arr))
        print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    print(np.int32(myIndex))

    # Grading
    grading = []
    for x in range(0, questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    print(grading)
    score = (sum(grading)/questions)*100  # Final Score
    print(score)

    # Displaying answers
    imgResult = imgWarpColored.copy()
    utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
    imgRawDrawing = np.zeros_like(imgWarpColored)
    utils.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)
    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade, str(int(score))+"%", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
    cv2.imshow("Grade", imgRawGrade)
    invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
              [imgResult, imgRawDrawing, imgInvWarp, imgFinal])
labels = [["Original", "Gray", "Blur", "Canny"],
          ["Contours", "Biggest Con", "Warp", "Threshold"],
          ["Result", "Raw Drawing", "Inv Warp", "Final"]]
imgStacked = utils.stackImages(imageArray, 0.3, labels)

cv2.imshow("Final Result", imgFinal)
cv2.imshow("Stacked Images", imgStacked)
cv2.waitKey(0)
