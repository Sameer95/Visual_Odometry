#include "feature.h"
#include <fstream>
using namespace cv;
using namespace std;

#define MAX_FRAME 2999
#define MIN_NUM_FEAT 2000
#define pi 3.14159265


double yaw,roll,pitch;
void y_p_r(cv::Mat& R)
{
                               //in radians
  yaw=atan ( R.at<double>(1,0)/R.at<double>(0,0));
  double cos_yaw,sin_yaw;
  cos_yaw=cos(yaw);
  sin_yaw=sin(yaw);
  double tp;
  tp=R.at<double>(0,0)*cos_yaw+R.at<double>(1,0)*sin_yaw;
  pitch=atan((-1*R.at<double>(2,0)) / tp);

  // roll=atan(R.at<double>(2,1)/R.at<double>(2,2));
  roll=atan( (R.at<double>(0,2)*sin_yaw - R.at<double>(1,2)*cos_yaw)/ (-R.at<double>(0,1)*sin_yaw + R.at<double>(1,1)*cos_yaw));
  // char wait;
  roll=roll*180/pi;
  yaw=yaw*180/pi;
  pitch=pitch*180/pi;
  // cout<<roll<<" "<<yaw<<" "<<pitch<<endl;
  // if(roll>2 || yaw>2 || pitch>2)
  // {
  //   // cin>>wait;
  //   waitKey(0);
  // }
}

void check_E(cv::Mat& E,cv::Mat& F,vector<Point2f>& points1,vector<Point2f>& points2,cv::Mat& K)
{
  Mat pt1=(Mat_<double>(3,1)<<0,0,1);
  Mat pt2=(Mat_<double>(3,1)<<0,0,1);
  int count=0;
  double highest=0;
  Mat temp,hold;
  for(int j=0;j<points1.size();j++)
  {
    pt1.at<double>(0,0)=points1[j].x;
    pt1.at<double>(0,1)=points1[j].y;
    pt2.at<double>(0,0)=points2[j].x;
    pt2.at<double>(0,1)=points2[j].y;

    hold=K.inv();
    temp=pt2.t()*hold.t()*E*hold*pt1;
    // temp=pt2.t()*F*pt1;
    if(temp.at<double>(0,0)>highest)
    {
      highest=temp.at<double>(0,0);
    }
    if(temp.at<double>(0,0) <= 1)
    {
      count++;
    }
  }
  // if(highest<10) 
  // {
  //   return 1;
  // }
  // else {return 0;}
  cout<<"total="<<points1.size()<<" & good pts="<<count<<" & highest="<<highest<<endl;
}
void check_E2(cv::Mat& img_1,cv::Mat& img_2,cv::Mat& R,cv::Mat& t,vector<Point2f> points1,vector<Point2f> points2)
{
  Mat pt1=(Mat_<double>(3,1)<<0,0,1);
  Mat pt2=(Mat_<double>(3,1)<<0,0,1); 
  int count=0;
  double highest=0;
  Mat temp;
  vector<DMatch> corresp(points1.size());
  for(int i = 0; i <points1.size(); ++i){
    corresp[i] = cv::DMatch(i, i, 0);
  }
  for(int j=0;j<points1.size();j++)
  {
    pt1.at<double>(0,0)=points1[j].x;
    pt1.at<double>(0,1)=points1[j].y;
    pt2.at<double>(0,0)=points2[j].x;
    pt2.at<double>(0,1)=points2[j].y;

    pt1=R*(pt1-t);

    points1[j].x=pt1.at<double>(0,0);
    points1[j].y=pt1.at<double>(0,1);
  }
  // cout<<points2[0]<<endl<<points1[0]<<endl;
  //draw
  Mat matches;
  vector<KeyPoint> points1_t, points2_t;
  points1.resize(100);
  points2.resize(100);
  corresp.resize(100);
  cout<<points1[0]<<endl<<points2[0]<<endl;
  KeyPoint::convert(points2,points2_t);
  KeyPoint::convert(points1,points1_t);
  namedWindow("Back_projection", WINDOW_AUTOSIZE );// Create a window for display.
  cv::drawKeypoints(img_2,points1_t,matches,Scalar::all(255),DrawMatchesFlags::DEFAULT);
  imshow( "Back_projection" , matches);
  waitKey(0);
  cv::drawKeypoints(img_2,points2_t,matches,Scalar::all(255),DrawMatchesFlags::DEFAULT);
  imshow( "Back_projection" , matches);
  waitKey(0);
  // cv::drawMatches(img_2,points1_t,img_2,points2_t,corresp,matches,Scalar::all(-1),Scalar::all(0),vector<char>(),DrawMatchesFlags::DEFAULT);
  // resize(matches,matches,Size(),0.6,0.6);
}
void convert(vector<Point2f>& points2,vector<Point2f>& points1,cv::Mat& K,int flag)
{
  if(flag==0)                               // camera point to image point
  {
    
    Mat tp1=(Mat_<double>(3,1) << 0,0,1);
    Mat tp2=(Mat_<double>(3,1) << 0,0,1);
    for(int i=0;i<points1.size();i++)
    {
      tp1.at<double>(0,0)=points1[i].x;
      tp1.at<double>(0,1)=points1[i].y;
      tp2.at<double>(0,0)=points2[i].x;
      tp2.at<double>(0,1)=points2[i].y;

      tp1=K*tp1;
      points1[i].x=tp1.at<double>(0,0);
      points1[i].y=tp1.at<double>(0,1);
      tp2=K*tp2;
      points2[i].x=tp2.at<double>(0,0);
      points2[i].y=tp2.at<double>(0,1);
    }
  }
  if(flag==1)                             // image point to camera point
  {
    Mat K_inv=K.inv();
    Mat tp1=(Mat_<double>(3,1) << 0,0,1);
    Mat tp2=(Mat_<double>(3,1) << 0,0,1); 
    for(int i=0;i<points1.size();i++)
    {
      tp1.at<double>(0,0)=points1[i].x;
      tp1.at<double>(0,1)=points1[i].y;
      tp2.at<double>(0,0)=points2[i].x;
      tp2.at<double>(0,1)=points2[i].y;

      tp1=K_inv*tp1;
      points1[i].x=tp1.at<double>(0,0);
      points1[i].y=tp1.at<double>(0,1);
      tp2=K_inv*tp2;
      points2[i].x=tp2.at<double>(0,0);
      points2[i].y=tp2.at<double>(0,1);
    } 
  }
}
void essen_mat(cv::Mat& img_1,cv::Mat& img_2,vector<Point2f>& points2,vector<Point2f>& points1, cv::Mat& R, cv::Mat& t, cv::Mat& K,cv::Point2d& pp)
{
  double focal1=K.at<double>(0,0);
  double focal2=K.at<double>(1,1);
  int E_ret;
  Mat F,E,mask,K_t;
  // convert(points2,points1,K,0);                      //convert from camera to image point (now points1 and points2 are image points)
  F = findFundamentalMat(points1,points2,CV_FM_RANSAC, 0.999, 1.0, mask);
  // check_F(E,F,points1,points2,K);                 // F looks fine .. satisfying x'Fx=0
  K_t=K.t();
  E= K_t*F*K;
  // convert(points2,points1,K,1);                     //convert back from image point to camera point
  // E_ret=
  // check_E(E,F,points1,points2,K);                   //check if E is right using epipolar equation
  
  SVD decomp = SVD(E);
  
  Mat U = decomp.u;

  Mat S(3, 3, CV_64F, Scalar(0));
  S.at<double>(0, 0) = decomp.w.at<double>(0, 0);
  S.at<double>(1, 1) = decomp.w.at<double>(0, 1);
  S.at<double>(2, 2) = decomp.w.at<double>(0, 2);

  Mat V = decomp.vt;

  Mat W(3, 3, CV_64F, Scalar(0));
  W.at<double>(0, 1) = -1;
  W.at<double>(1, 0) = 1;
  W.at<double>(2, 2) = 1;
  
  rec_pose(E, points1, points2, R, t, focal1,focal2, pp, mask);
  check_E2(img_1,img_2,R,t,points1,points2);
  cout<<E<<endl;
  // return E_ret;
}

int main( int argc, char** argv ) {

  Mat img_1, img_2;
  Mat R_f, t_f; //the final rotation and tranlation vectors containing the 


  // original image : 1616x616 px
  // rotated image : 616x1616 px

  double focal1 = 408.3971/2;                      //intrinsic parameters.
  double focal2 = 408.3971/2;                      //intrinsic parameters.
  double focal=focal1;
  cv::Point2d pp(380-192.599,303.438);            //for rot
  Mat K = (Mat_<double>(3,3) << focal1, 0, pp.x, 0, focal2,pp.y, 0, 0, 1);

  double scale = 1.00;
  char filename1[200];
  char filename2[200];
  sprintf(filename1, "/path_to_dataset/image%04d.jpg", 80);
  sprintf(filename2, "/path_to_dataset/image%04d.jpg", 81);

  char text[100];
  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;  
  cv::Point textOrg(10, 50);

  //read the first two frames from the dataset
  Mat img_1_c = imread(filename1);
  Mat img_2_c = imread(filename2);
  Mat keyp,re_keyp;

  if ( !img_1_c.data || !img_2_c.data ) { 
    std::cout<< "Error" << std::endl; return -1;
  }

  // we work with grayscale images
  cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
  cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);
  // feature detection, tracking
  vector<Point2f> points1_samp,points2_samp,points1, points2;        //vectors to store the coordinates of the feature points
  featureDetection2(img_1, points1);        //detect features in img_1
  // points1=points1_orig;

  vector<uchar> status;
  featureTracking(img_1,img_2,points1,points2, status); //track those features to img_2
  rand_sample(points1,points2,points1_samp,points2_samp);

  vector<KeyPoint> points1_t, points2_t;        //vectors to store the coordinates of the feature points
  KeyPoint::convert(points2,points2_t);
  KeyPoint::convert(points1,points1_t);
  vector<DMatch> corresp(points1_t.size());

  Mat matches;
  for(int i = 0; i <points1_t.size(); ++i){
    corresp[i] = cv::DMatch(i, i, 0);
  }

  cv::drawMatches(img_1,points1_t,img_2,points2_t,corresp,matches,Scalar::all(-1),Scalar::all(-1),vector<char>(),DrawMatchesFlags::DEFAULT);
  namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
  resize(matches,matches,Size(),0.4,0.4);
  imshow( "Road facing camera" , matches);
  waitKey(0); 
  
  
  // Mat R, t;
  // int E_consistency;
  // E_consistency=
  // essen_mat(img_1,img_2,points2,points1,R,t,K,pp);      // improve cases for checking E_consistency    
                                                          // can do somenthing if E_consistency =0 i.e x.t()*E*x !=0 
  
  Mat E,R, t, mask;
  Mat F,tem;
  // convert(points1,points2,K,1);
  E = findEssentialMat(points1, points2, focal, pp, RANSAC, 0.999, 1.0, mask);         // for comparing when scaling factor is not there
  recoverPose(E, points1, points2, R, t, focal, pp, mask);

  // debug(t);
 
  Mat prevImage = img_2;
  Mat currImage;
  vector<Point2f> prevFeatures = points2;
  vector<Point2f> currFeatures,prevFeatures_samp,currFeatures_samp;

  char filename[100];

  R_f = R.clone();
  t_f = t.clone();
  clock_t begin = clock();

  namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.
  // Mat aa;
  Mat traj = Mat::zeros(1000,1000, CV_8UC3);

  int sz1=220;
  int sz2=180;
  int sz3=118;
  int sz4=125;
  int sz5=246;
  int sz6=132;
  int sz7=606;
  int sz8=307;
  vector<int> ar1(sz1);
  vector<int> ar2(sz2);
  vector<int> ar3(sz3);
  vector<int> ar4(sz4);
  vector<int> ar5(sz5);
  vector<int> ar6(sz6);
  vector<int> ar7(sz7); 
  vector<int> ar8(sz8); 
  
  for(int i=0;i<sz1;i++)
  {
    ar1[i]=80+i;
  }
  for(int i=0;i<sz2;i++)
  {
    ar2[i]=410+i;
  }
  for(int i=0;i<sz3;i++)
  {
    ar3[i]=646+i;
  }
  for(int i=0;i<sz4;i++)
  {
    ar4[i]=847+i;
  }
  for(int i=0;i<sz5;i++)
  {
    ar5[i]=1004+i;
  }
  for(int i=0;i<sz6;i++)
  {
    ar6[i]=1270+i;
  }
  for(int i=0;i<sz7;i++)
  {
    ar7[i]=1994+i;
  }
  for(int i=0;i<sz8;i++)
  {
    ar8[i]=2693+i;
  }

  int sz=sz1+sz2+sz3+sz4+sz5+sz6+sz7+sz8;
  vector<int> ar_all;
  ar_all.reserve(sz);
  ar_all.insert( ar_all.end(), ar1.begin(), ar1.end() );
  ar_all.insert( ar_all.end(), ar2.begin(), ar2.end() );
  ar_all.insert( ar_all.end(), ar3.begin(), ar3.end() );
  ar_all.insert( ar_all.end(), ar4.begin(), ar4.end() );
  ar_all.insert( ar_all.end(), ar5.begin(), ar5.end() );
  ar_all.insert( ar_all.end(), ar6.begin(), ar6.end() );
  ar_all.insert( ar_all.end(), ar7.begin(), ar7.end() );
  ar_all.insert( ar_all.end(), ar8.begin(), ar8.end() );
  
  int breaker;
// #############################################################      loop      ########################################################################
  for(int numFrame=2; ar_all[numFrame] < MAX_FRAME; numFrame=numFrame+1) {
    sprintf(filename, "/path_to_dataset/image%04d.jpg", ar_all[numFrame]);
    cout << ar_all[numFrame] << endl;
    Mat currImage_c = imread(filename);
    if(!currImage_c.data)
    {
      continue;
    }
    cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
    vector<uchar> status;
    prevFeatures.clear();                                                 
    currFeatures.clear();
    if(ar_all[numFrame]==1004)
    {
    sprintf(filename, "/path_to_dataset/image%04d.jpg", ar_all[numFrame]-1);  
    Mat prevImage_c=imread(filename);
    cvtColor(prevImage_c, prevImage, COLOR_BGR2GRAY);
    }

    featureDetection2(prevImage,prevFeatures);
    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
    
    rand_sample(prevFeatures,currFeatures,prevFeatures_samp,currFeatures_samp);

    if(currFeatures.size()<20)
    {
      vector<uchar> status;
      featureDetection2(prevImage,prevFeatures);
      featureTracking(prevImage,currImage,prevFeatures,currFeatures,status);
      rand_sample(prevFeatures,currFeatures,prevFeatures_samp,currFeatures_samp);
    }

    // convert(currFeatures,prevFeatures,K,1);
    E = findEssentialMat( prevFeatures,currFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E,prevFeatures,currFeatures, R, t, focal, pp, mask);
    // convert(currFeatures,prevFeatures,K,0);
    // essen_mat(currFeatures,prevFeatures,R,t,K,pp);
    // check_E(E,F,currFeatures,prevFeatures,K);

    // Mat hold_image=currImage.clone();
    // epilines(prevFeatures_samp,currFeatures_samp,E,K,prevImage,hold_image);
      

    // cout<<R.at<double>(0,0)<<" "<<R.at<double>(0,1)<<" "<<R.at<double>(1,0)<<" "<<R.at<double>(1,1)<<" ";
    // cout<<R.at<double>(2,0)<<" "<<R.at<double>(2,1)<<" "<<R.at<double>(2,2)<<" "<<R.at<double>(1,2)<<" "<<R.at<double>(0,2)<<endl;
    // cout<<R<<endl;
    // y_p_r(R);
  
    // so3_rot2rph(R,eulerAngles);
    // cout<<eulerAngles<<endl;                    //roll,pitch,yaw
    
    Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);
    Mat mtchs;

   // for(int i=0;i<prevFeatures.size();i++) {   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
   //    prevPts.at<double>(0,i) = prevFeatures.at(i).x;
   //    prevPts.at<double>(1,i) = prevFeatures.at(i).y;

   //    currPts.at<double>(0,i) = currFeatures.at(i).x;
   //    currPts.at<double>(1,i) = currFeatures.at(i).y;
   //  }

    // scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));        // update when using kalman filter
    scale=1;

    // if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

      // if(ar_all[numFrame]!=862)
      // {
        R_f = R*R_f;
        t_f = t_f + scale*(R_f*t);
      // }

    // }
    
    // else {
    //  cout << "scale below 0.1, or incorrect translation" << endl;
    // }
    
  
    prevImage = currImage.clone();
    prevFeatures = currFeatures;

    int x = int(t_f.at<double>(0)) + 800;
    int y = int(t_f.at<double>(2)) + 550;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

    rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    sprintf(text, "Coordinates: x = %02fm z = %02fm", t_f.at<double>(2), t_f.at<double>(0));
    y_p_r(R);
    // myfile<<t_f.at<double>(0)<<" "<<t_f.at<double>(2)<<" "<<roll<<" "<<pitch<<" "<<yaw<<endl;
    putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
    // imshow( "Road facing camera", aa );
    KeyPoint::convert(prevFeatures,points1_t);             //covert 2d pts to keypts
    KeyPoint::convert(currFeatures,points2_t);
    vector<DMatch> corresp(points1_t.size());

    for(int i = 0; i <points1_t.size(); ++i){         // creating DMatch vector
      corresp[i] = cv::DMatch(i, i, 0);
    }
    Mat prevImage_t,currImage_t;
    // rotate(prevImage,-90,prevImage_t);
    // rotate(currImage,-90,currImage_t);
    cv::drawMatches(prevImage,points1_t,currImage,points2_t,corresp,matches,Scalar::all(-1),Scalar::all(-1),vector<char>(),DrawMatchesFlags::DEFAULT);
    resize(matches,matches,Size(),0.75,0.75);
    imshow( "Road facing camera" , prevImage);
    imshow( "Trajectory", traj );
    // waitKey(1);

    if(waitKey(30)>=0) {breaker=waitKey(-1);if(breaker==1048603) break;}
  }
  waitKey(0);
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total time taken: " << elapsed_secs << "s" << endl;

  // cout << R_f << endl;
  // cout << t_f << endl;

  return 0;
}
