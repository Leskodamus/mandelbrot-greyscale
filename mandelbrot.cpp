#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

/**
 * Ressources: https://docs.opencv.org/4.x/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html
*/

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

    Scale(const Mat &img, const Area &area) :
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

/* Parallelly generate mandelbrot set image with OpenCV */
void mandelbrot_parallel(Mat &img, Area &area, const unsigned int limit = 200) {
    Scale scale(img, area);
    parallel_for_(Range(0, img.rows*img.cols), [&](const Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            int row = i / img.cols;
            int col = i % img.cols;

            double real = col / scale.x + area.x_min;
            double imag = row / scale.y + area.y_min;
            std::complex<double> c(real, imag);
            int value = mandelbrot_greyscale(c, limit);
            img.ptr<uchar>(row)[col] = static_cast<uchar>(value);
        }
    });
}

bool is_number(const std::string &s) {
    if (s.empty()) return false;
    for (char const &ch : s) {
        if (std::isdigit(ch) == 0)
            return false;
    }
    return true;
}

void usage(void) {
    std::cout << "Usage: ./mandelbrot [OPTION]...\n"
        << "Options:\n\t-i, --iteration\t\tnumber of iterations\n"
        << "\t-r, --res\t\timage size <width>x<height>; e.g. '--res 400x600'\n"
        << "\t-p, --parallel\t\trun generation multithreaded"<< std::endl;
}

int main(const int argc, const char* argv[]) {
    bool is_parallel = false;
    int limit = 200;
    int x_pixel = 400, y_pixel = 600;

    /* Parse args */
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "--iteration") {
            if (i+1 < argc) {
                if (is_number(argv[i+1])) {
                    limit = std::stoi(argv[++i]);
                } else {
                    std::cerr << "Iteration must be a positive integer." << std::endl;
                    usage();
                    return 1;
                }
            } else {
                std::cerr << "-i or --iteration option requires one argument." << std::endl;
                usage();
                return 1;
            }
        } else if (arg == "-r" || arg == "--res") {
            if (i+1 < argc) {
                std::string res = argv[++i];
                std::string res_y = res.substr(0, res.find("x"));
                std::string res_x = res.substr(res.find("x")+1, res.size());
                if (is_number(res_x)) {
                    x_pixel = std::stoi(res_x);
                } else {
                    std::cerr << "X resolution must be a positive integer." << std::endl;
                    usage();
                    return 1;
                }
                if (is_number(res_y)) {
                    y_pixel = std::stoi(res_y);
                } else {
                    std::cerr << "Y resolution must be a positive integer." << std::endl;
                    usage();
                    return 1;
                }
            } else {
                std::cerr << "-r or --res option requires one argument." << std::endl;
                usage();
                return 1;
            }
        } else if (arg == "-p" || arg == "--parallel") {
            is_parallel = true;
        } else if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        }
    }

    Mat img(x_pixel, y_pixel, CV_8U);
    Area area(-2.1, 0.6, -1.2, 1.2);

    if (is_parallel)
        mandelbrot_parallel(img, area, limit);
    else
        mandelbrot(img, area, limit);

    std::string fname = "mandelbrot_" + std::to_string(y_pixel) + 
        "x" + std::to_string(x_pixel) + "_" + std::to_string(limit) + ".png";
    imwrite(fname, img);

    return 0;
}

