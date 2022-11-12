#include <iostream>
#include <vector>
#include <complex>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

using namespace cv;

/* Area domain for points */
struct Area {
    double x_min;
    double x_max;
    double y_min;
    double y_max;

    Area(double _x_min, double _x_max, double _y_min, double _y_max) :
        x_min(_x_min), x_max(_x_max),
        y_min(_y_min), y_max(_y_max) {}
};

/* Scale for Area to Mat */
struct Scale {
    double x;
    double y;
    
    Scale() = default;

    Scale(Mat &img, Area &area) :
        x(img.cols / (area.x_max - area.x_min)), 
        y(img.rows / (area.y_max - area.y_min)) {}
};

/**
 * Escape Time Algorithm
 * formula: z(n+1) = z(n)^2 + c
*/
int escape_time(const std::complex<double> &c, const int limit) {
    std::complex<double> z = c;
    for (int i = 0; i < limit; ++i) {
        if (std::norm(z) > 4.0) return i;
        z = z*z+c;
    }
    return limit;
}

/* Get greyscale value 0-255 */
int mandelbrot_greyscale(const std::complex<double> &c, const int limit) {
    int value = escape_time(c, limit);
    if (limit - value == 0) return 0;
    return cvRound(sqrt(value / static_cast<double>(limit)) * 255);
}

/* Sequentially generate mandelbrot set image */
void mandelbrot(Mat &img, Area &area, const unsigned int limit = 200) {
    Scale scale(img, area);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            double real = j / scale.x + area.x_min;
            double imag = i / scale.y + area.y_min;
            std::complex<double> c(real, imag);
            int value = mandelbrot_greyscale(c, limit);
            img.ptr<uchar>(i)[j] = static_cast<uchar>(value);
        }
    }
}

int main(void) {
    Mat img(400, 600, CV_8U);
    Area area(-2.1, 0.6, -1.2, 1.2);
    mandelbrot(img, area);
    imwrite("mandelbrot.png", img);
    return 0;
}

