#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

double distance_from_edge( const cv::Mat& distance_img, const cv::Point2d w)
{
    if ( w.y >= distance_img.rows || w.x >= distance_img.cols ) {
        double minVal; 
        double maxVal; 
        Point minLoc; 
        Point maxLoc;
        cv::minMaxLoc( distance_img, &minVal, &maxVal, &minLoc, &maxLoc );
        return (pow(w.y,2)+pow(w.x,2))*maxVal;
    }
    return distance_img.at<double>(w.y,w.x);
}

typedef std::vector<cv::Point2d> W_t;

double vector_norm2(const W_t::value_type& p)
{
    return sqrt(pow(p.x,2) + pow(p.y,2));
}

double space(const W_t& W, const size_t& N, const size_t& n)
{
    double a = 0;
    for( size_t i = 1; i<= N; i++ ) {
        a += vector_norm2(W[i]-W[i-1]);
    }
    a /= N;
    return -pow(a-vector_norm2(W[n]-W[n-1]),2);
}

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    if(! image.data )                              
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "image", WINDOW_AUTOSIZE );
    imshow( "image", image );                  

    cv::Mat edges;
    cv::Canny(image,edges,100,200);

    namedWindow( "edges", WINDOW_AUTOSIZE );
    imshow( "edges", edges );                  

    cv::Mat distance(edges.rows,edges.cols,CV_32FC1);
    cv::distanceTransform(255-edges,distance,  CV_DIST_C,CV_DIST_MASK_PRECISE);

    assert(32==4*sizeof(double));

    cv::Mat normalized_dist;
    cv::normalize(distance, normalized_dist, 0.0, 1.0, NORM_MINMAX);

    namedWindow( "distance", WINDOW_AUTOSIZE );
    imshow( "distance", normalized_dist );    

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles( edges, circles, CV_HOUGH_GRADIENT, 1, 10, 200, 25, 0, 0 );

    std::cout << circles.size() << "\n";

    //for( size_t i = 0; i < circles.size(); i++ )
    cv::Mat hough = image.clone();
    // for( size_t i = 0; i < 1; i++ )
    // {
    //    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    //    int radius = cvRound(circles[i][2]);
    //    circle( hough, center, 3, Scalar(255,255,255));
    //    circle( hough, center, radius, Scalar(255,255,255));
    // }    

    
    const size_t selected_circle = 1;
    cv::Point2d w_center(circles[selected_circle][0],circles[selected_circle][1]);
    double radius = circles[selected_circle][2];
    const size_t N = 10;
    W_t W(N+1);
    for ( size_t i = 1; i <= N; i++ ) {
        const double a = i*2*M_PI/N;
        W[i].x = w_center.x + radius*cos(a);
        W[i].y = w_center.y + radius*sin(a);
    }
    W[0] = W[N];

    for ( size_t i = 1; i <= N; i++ ) {
        circle( hough, W[i], 3, Scalar(255,255,255));
    }

    namedWindow( "circles", WINDOW_AUTOSIZE );
    imshow( "circles", hough );

    for ( size_t i = 1; i <= N; i++ ) {
        std::cout << space(W,N,i) << "\n";
    }

    waitKey(0);                                
    return 0;
}
