#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <vector>
 
using namespace cv;
using namespace std;
 
// Function to calculate the centroid of a contour
Point2f getCentroid(const vector<Point> &contour)
{
    Moments m = moments(contour, false);
    Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
    return centroid;
}
 
// Structure to hold bounding box data
struct BoundingBox
{
    Rect box;
    Point2f centroid;
    int id;
};
 
 
void carDetectionAndCounting()
{
    // Open video file
    VideoCapture cap("HSCC Interstate Highway Surveillance System - TEST VIDEO.mp4");
    if (!cap.isOpened())
    {
        cout << "Error: Could not open video file." << endl;
        return ;
    }
 
    // Create Background Subtractor
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
 
    Mat frame, fgMask;
    map<int, BoundingBox> objects;
    int objectID = 0;
    int carCountLeft = 0;
    int carCountRight = 0;
    int carCountTotal = 0;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;
 
        // Apply background subtraction
        pBackSub->apply(frame, fgMask);
 
        // Create a mask for the region of interest
        Mat mask = Mat::zeros(fgMask.size(), fgMask.type());
        Point pts[6] = {
            Point(frame.cols * 0, frame.rows * 1),
            Point(frame.cols * 0, frame.rows * 0.45),
            Point(frame.cols * 0.3, frame.rows * 0.3),
            Point(frame.cols * 0.73, frame.rows * 0.3),
            Point(frame.cols * 1, frame.rows * 0.45),
            Point(frame.cols * 1, frame.rows * 1)};
 
        fillConvexPoly(mask, pts, 6, Scalar(255));
 
        // Apply the mask to the edge image
        bitwise_and(fgMask, mask, fgMask);
        imshow("fgMask with ROI", fgMask);
 
        // Apply threshold to create binary image
        threshold(fgMask, fgMask, 200, 255, THRESH_BINARY);
 
        // Morphological operations to improve blob detection
        morphologyEx(fgMask, fgMask, MORPH_OPEN, Mat(), Point(-1, -1), 2);
        morphologyEx(fgMask, fgMask, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
 
        // Find contours
        vector<vector<Point>> contours;
        findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
 
        // Track objects
        map<int, BoundingBox> newObjects;
        for (const auto &contour : contours)
        {
            if (contourArea(contour) < 500)
                continue; // Filter small contours
            Rect boundingBox = boundingRect(contour);
            Point2f centroid(boundingBox.x + boundingBox.width / 2.0, boundingBox.y + boundingBox.height / 2.0);
 
            bool matched = false;
            for (auto &obj : objects)
            {
                if (norm(obj.second.centroid - centroid) < 70)
                { // Match objects based on centroid distance
                    BoundingBox bbox = {boundingBox, centroid, obj.first};
                    newObjects[obj.first] = bbox;
                    matched = true;
                    break;
                }
            }
            if (!matched)
            {
                BoundingBox bbox = {boundingBox, centroid, objectID++};
                newObjects[bbox.id] = bbox;
            }
        }
 
        // Define lines for left and right lanes
        int midY = frame.rows / 2;
        int leftLaneY = midY + 20;  // Adjust this value based on the lane position
        int rightLaneY = midY + 30; // Adjust this value based on the lane position
 
        // Count cars for the left lane
        for (const auto &obj : objects)
        {
            if (obj.second.centroid.y < leftLaneY && newObjects.find(obj.first) != newObjects.end() && newObjects[obj.first].centroid.y >= leftLaneY)
            {
                carCountLeft++;
                carCountTotal++;
            }
        }
 
        // Count cars for the right lane
        for (const auto &obj : objects)
        {
            if (obj.second.centroid.y > rightLaneY && newObjects.find(obj.first) != newObjects.end() && newObjects[obj.first].centroid.y <= rightLaneY)
            {
                carCountRight++;
                carCountTotal++;
            }
        }
 
        // Update objects
        objects = newObjects;
 
        // Draw results
        for (const auto &obj : objects)
        {
            rectangle(frame, obj.second.box, Scalar(0, 0, 255), 2);
            circle(frame, obj.second.centroid, 5, Scalar(0, 255, 0), -1);
            putText(frame, to_string(obj.first), obj.second.centroid, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
        }
 
        // Draw the lane lines
        line(frame, Point(0, leftLaneY), Point(frame.cols / 2 - 90, leftLaneY), Scalar(255, 0, 0), 2);
        line(frame, Point(frame.cols - frame.cols / 2 + 100, rightLaneY), Point(frame.cols, rightLaneY), Scalar(0, 255, 0), 2);
 
        // Car count on the screen
        putText(frame, "Left Car : " + to_string(carCountLeft), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        putText(frame, "Right Car: " + to_string(carCountRight), Point(frame.cols /2 + 120, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        putText(frame, "Total : " + to_string(carCountTotal), Point(frame.cols / 2 - 60, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
 
        // Display the results
        imshow("Frame", frame);
        imshow("FG Mask", fgMask);
 
        if (waitKey(30) == 27)
            break; // Stop if 'Esc' key is pressed
    }
 
    cout << "Left Lane Car Count: " << carCountLeft << endl;
    cout << "Right Lane Car Count: " << carCountRight << endl;
    cout << "Total Car Count: " << carCountTotal << endl;
    cap.release();
    destroyAllWindows();
}
 
 
 
 
int main(int argc, char const *argv[])
{
    
    carDetectionAndCounting();
    return 0;
}