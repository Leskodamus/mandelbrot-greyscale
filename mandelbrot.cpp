#include <iostream>
#include <vector>
#include <complex>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

/**
 * formula iteration: z(n+1) = z(n)^2 + c
*/

int escape_time(const std::complex<double> &c, const int limit) {
    std::complex<double> z = c;
    for (int i = 0; i < limit; ++i) {
        if (std::norm(z) > 4.0) return i;
        z = z*z+c;
    }
    return limit;
}

int mandelbrot_greyscale(std::complex<double> &c, const int limit = 200) {
    int value = escape_time(c, limit);
    if (limit - value == 0) return 0;
    return cvRound(sqrt(value / static_cast<double>(limit)) * 255);
}

void mandelbrot(Mat &img, const double x1, const double y1,
                const double scaleX, const double scaleY) 
{
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            double x0 = j / scaleX + x1;
            double y0 = i / scaleY + y1;
            std::complex<double> c(x0, y0);
            int value = mandelbrot_greyscale(c);
            img.ptr<uchar>(i)[j] = static_cast<uchar>(value);
        }
    }
}

int main(void) {
    Mat img(400, 600, CV_8U);
    double x1 = -2.1, x2 = 0.6;
    double y1 = -1.2, y2 = 1.2;
    double scaleX = img.cols / (x2 - x1);
    double scaleY = img.rows / (y2 - y1);

    mandelbrot(img, x1, y1, scaleX, scaleY);
    imwrite("mandelbrot.png", img);

    return 0;
}

