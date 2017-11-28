#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

void rand_sample(vector<Point2f>& points1,vector<Point2f>& points2,vector<Point2f>& points1_samp,vector<Point2f>& points2_samp)
{
  while(!points1_samp.empty())
  {
    points1_samp.pop_back();
    points2_samp.pop_back();
  }

  cv::RNG rng(5);
  int hold;
  for(int i=0;i<40;i++)
  {
    hold=rng.uniform(0,points1.size());
    points1_samp.push_back(points1[hold]);
    points2_samp.push_back(points2[hold]);
  }
  // cout<<points1_orig.size()<<endl;
}
void drawlines(cv::Mat img1,cv::Mat img2,vector<Point3f>& lines,vector<Point2f>& pts1,vector<Point2f>& pts2)
{
  namedWindow( "epipolar", WINDOW_AUTOSIZE );
  int r,c;
  r=img1.rows;
  c=img1.cols;
  Point3f l;
  Point2f tp1,tp2;
  Point p0,p1;
  for(int i=0;i<pts1.size();i++)
  {
    l=lines[i];
    tp1=pts1[i];
    tp2=pts2[i];
    p0.x=0;
    p0.y=(-l.z/l.y);
    p1.x=c;
    p1.y=-(l.z+l.x*c)/l.y;
    line(img1,p0,p1,Scalar::all(-1),1);
    // circle(img1, tp1 ,1, CV_RGB(255,0,0), 2);
    circle(img1, tp2 ,1, CV_RGB(255,0,0), 2);
    // cout<<i<<endl;
  }
    imshow( "epipolar" , img1);
    waitKey(0);
}
void epilines(vector<Point2f>& pts1,vector<Point2f>& pts2,cv::Mat E,cv::Mat K,cv::Mat img1,cv::Mat img2)
{
 vector<Point3f> lines;
 Mat F,tp;
 tp=K.inv();
 F=tp.t()*E*tp;
 computeCorrespondEpilines(pts1,1,F,lines);
 drawlines(img2,img1,lines,pts2,pts1);
}

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 

//this function automatically gets rid of points for which tracking fails

  vector<float> err;					
  Size winSize=Size(21,21);																								
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame

  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  
      Point2f pt = points2.at(i- indexCorrection);
      if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))  {
          if((pt.x<0)||(pt.y<0))  {
            status.at(i) = 0;
          }
          points1.erase (points1.begin() + (i - indexCorrection));
          points2.erase (points2.begin() + (i - indexCorrection));
          indexCorrection++;
      }

     }
  
  // cout<<points2.size()<<" "<<points1.size()<<" ";

}
void featureDetection2(Mat img_1, vector<Point2f>& points1)  {   //uses FAST as of now, modify parameters as necessary
  vector<KeyPoint> keypoints_1;
  cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
  f2d->detect(img_1, keypoints_1);
  KeyPoint::convert(keypoints_1, points1, vector<int>());

}

void featureDetection(Mat img_1, vector<Point2f>& points1)	{   //uses FAST as of now, modify parameters as necessary
  vector<KeyPoint> keypoints_1;
  int fast_threshold = 10;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold);
  KeyPoint::convert(keypoints_1, points1, vector<int>());

}

int rec_pose(InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
                     OutputArray _t, double focal1,double focal2, Point2d pp, InputOutputArray _mask)
{
  Mat points1, points2;
    _points1.getMat().copyTo(points1);
    _points2.getMat().copyTo(points2);

    int npoints = points1.checkVector(2);
    CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
                              points1.type() == points2.type());

    if (points1.channels() > 1)
    {
        points1 = points1.reshape(1, npoints);
        points2 = points2.reshape(1, npoints);
    }
    points1.convertTo(points1, CV_64F);
    points2.convertTo(points2, CV_64F);

    points1.col(0) = (points1.col(0) - pp.x) / focal1;
    points2.col(0) = (points2.col(0) - pp.x) / focal1;
    points1.col(1) = (points1.col(1) - pp.y) / focal2;
    points2.col(1) = (points2.col(1) - pp.y) / focal2;

    points1 = points1.t();
    points2 = points2.t();

    Mat R1, R2, t;
    decomposeEssentialMat(E, R1, R2, t);
    Mat P0 = Mat::eye(3, 4, R1.type());
    Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
    P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
    P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
    P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
    P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

    // Do the cheirality check.
    // Notice here a threshold dist is used to filter
    // out far away points (i.e. infinite points) since
    // there depth may vary between postive and negtive.
    double dist = 50.0;
    Mat Q;
    triangulatePoints(P0, P1, points1, points2, Q);
    Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask1 = (Q.row(2) < dist) & mask1;
    Q = P1 * Q;
    mask1 = (Q.row(2) > 0) & mask1;
    mask1 = (Q.row(2) < dist) & mask1;

    triangulatePoints(P0, P2, points1, points2, Q);
    Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask2 = (Q.row(2) < dist) & mask2;
    Q = P2 * Q;
    mask2 = (Q.row(2) > 0) & mask2;
    mask2 = (Q.row(2) < dist) & mask2;

    triangulatePoints(P0, P3, points1, points2, Q);
    Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask3 = (Q.row(2) < dist) & mask3;
    Q = P3 * Q;
    mask3 = (Q.row(2) > 0) & mask3;
    mask3 = (Q.row(2) < dist) & mask3;

    triangulatePoints(P0, P4, points1, points2, Q);
    Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask4 = (Q.row(2) < dist) & mask4;
    Q = P4 * Q;
    mask4 = (Q.row(2) > 0) & mask4;
    mask4 = (Q.row(2) < dist) & mask4;

    mask1 = mask1.t();
    mask2 = mask2.t();
    mask3 = mask3.t();
    mask4 = mask4.t();

    // If _mask is given, then use it to filter outliers.
    if (!_mask.empty())
    {
        Mat mask = _mask.getMat();
        CV_Assert(mask.size() == mask1.size());
        bitwise_and(mask, mask1, mask1);
        bitwise_and(mask, mask2, mask2);
        bitwise_and(mask, mask3, mask3);
        bitwise_and(mask, mask4, mask4);
    }
    if (_mask.empty() && _mask.needed())
    {
        _mask.create(mask1.size(), CV_8U);
    }

    CV_Assert(_R.needed() && _t.needed());
    _R.create(3, 3, R1.type());
    _t.create(3, 1, t.type());

    int good1 = countNonZero(mask1);
    int good2 = countNonZero(mask2);
    int good3 = countNonZero(mask3);
    int good4 = countNonZero(mask4);

    if (good1 >= good2 && good1 >= good3 && good1 >= good4)
    {
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask1.copyTo(_mask);
        return good1;
    }
    else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
    {
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask2.copyTo(_mask);
        return good2;
    }
    else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
    {
        t = -t;
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask3.copyTo(_mask);
        return good3;
    }
    else
    {
        t = -t;
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask4.copyTo(_mask);
        return good4;
    }
}
