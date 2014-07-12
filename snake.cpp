// See section "17.2 Snakes" in 
//
// @BOOK{princeCVMLI2012,
// author = {Prince, S.J.D.},
// title= {{Computer Vision: Models Learning and Inference}},
// publisher = {{Cambridge University Press}},
// year = 2012}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <string>

std::string getImgType(int imgTypeInt)
{
    int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)

    int enum_ints[] =       {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
                             CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
                             CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
                             CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                             CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
                             CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                             CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

    std::string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
                             "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
                             "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                             "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                             "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                             "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                             "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};

    for(int i=0; i<numImgTypes; i++)
    {
        if(imgTypeInt == enum_ints[i]) return enum_strings[i];
    }
    return "unknown image type";
}

typedef std::vector<cv::Point2d> W_t;
typedef float distance_t;

template< typename T >
struct CPP_TYPE_TO_OPENCV_IMG_TYPE
{
    CPP_TYPE_TO_OPENCV_IMG_TYPE() { assert(32==8*sizeof(distance_t)); }
};
template<> struct CPP_TYPE_TO_OPENCV_IMG_TYPE<distance_t> { enum { type = CV_32FC1 }; };


double distance_from_edge( const cv::Mat& distance_img, const W_t::value_type& w)
{
    if ( w.y >= distance_img.rows || w.x >= distance_img.cols ) {
        double minVal; 
        double maxVal; 
        cv::Point minLoc; 
        cv::Point maxLoc;
        cv::minMaxLoc( distance_img, &minVal, &maxVal, &minLoc, &maxLoc );
        std::cout << "rotto!!!";
        return (pow(w.y,2)+pow(w.x,2))*maxVal;
    }
    return distance_img.at<distance_t>(w.y,w.x);
}

double vector_norm2(const W_t::value_type& p)
{
    return sqrt(pow(p.x,2) + pow(p.y,2));
}

// Formula 17.5
double space(const W_t& W, const size_t& N, const size_t& n)
{
    double consecutive_distance_average = 0;
    for( size_t i = 1; i<= N; i++ ) {
        consecutive_distance_average += vector_norm2(W[i]-W[i-1]);
    }
    consecutive_distance_average /= N;
    double result = -pow(consecutive_distance_average-vector_norm2(W[n]-W[n-1]),2);
    std::cout << "space\t" << result << "\n";
    return result;
}

// Formula 17.6
double curve(const W_t& W, const size_t& n)
{
    double result = -pow(vector_norm2(W[n-1]-2*W[n]+W[n+1]),2);
    std::cout << "curve\t" << result << "\n";
    return result;
}

// Formula 17.4
double prior(const W_t& W, const size_t& N, const double& alpha, const double& beta)
{
    double result = 1;
    for ( size_t n = 1; n <= N; n++ ) {
        result *= exp(alpha*space(W,N,n)+beta*curve(W,n));
    }
    std::cout << "prior\t" << result << "\n";
    return result;
}

// Formula 17.3
double likelihood(const cv::Mat& distance_img, const W_t& W, const size_t& N)
{
    double result = 1;
    for ( size_t n = 1; n <= N; n++ ) {
        double d = distance_from_edge(distance_img,W[n]);
        result *= exp(-pow(d,2));;
        std::cout << "likelihood d\t" << d << "\t" << -pow(d,2) << "\n";
    }
    return result;
}

// Formula 17.7
double cost(const cv::Mat& distance_img, const W_t& W, const size_t& N,const double& alpha, const double& beta)
{
    return log(likelihood(distance_img,W,N)) + log(prior(W,N,alpha,beta));
}

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     std::cout <<" Usage: display_image ImageToLoadAndDisplay" << "\n";
     return -1;
    }

    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    if(! image.data )                              
    {
        std::cout <<  "Could not open or find the image" << "\n";
        return -1;
    }

    //cv::namedWindow( "image", cv::WINDOW_AUTOSIZE );
    cv::namedWindow( "image" );
    cv::imshow( "image", image );                  

    cv::Mat edges;
    cv::Canny(image,edges,100,200);
    edges = 255-edges;

    cv::namedWindow( "edges", cv::WINDOW_AUTOSIZE );
    cv::imshow( "edges", edges );                  

    cv::Mat distance(edges.rows,edges.cols,CPP_TYPE_TO_OPENCV_IMG_TYPE<distance_t>::type);
    
    cv::distanceTransform(edges,distance, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    std::ostringstream oss;
    oss << distance;
    std::cout << getImgType(distance.type()) << oss.str().substr(0,70) << "\n";

    cv::Mat normalized_dist;
    cv::normalize(distance.clone(), normalized_dist, 0.0, 1.0, cv::NORM_MINMAX);

    cv::namedWindow( "distance", cv::WINDOW_AUTOSIZE );
    cv::imshow( "distance", normalized_dist );    


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
    W_t W(N+2);
    for ( size_t i = 1; i <= N; i++ ) {
        const double a = i*2*M_PI/N;
        W[i].x = w_center.x + radius*cos(a);
        W[i].y = w_center.y + radius*sin(a);
    }
    W[0] = W[N];
    W[N+1] = W[1];

    for ( size_t i = 1; i <= N; i++ ) {
        circle( hough, W[i], 3, cv::Scalar(255,255,255));
    }

    cv::namedWindow( "circles", cv::WINDOW_AUTOSIZE );
    cv::imshow( "circles", hough );

    std::cout << prior(W,N,1,1) << "\n";
    std::cout << likelihood(distance,W,N) << "\n";

    cv::waitKey(0);                                
    return 0;
}
