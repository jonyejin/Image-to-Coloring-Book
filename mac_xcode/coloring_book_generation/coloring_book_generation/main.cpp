/**
 * \file      CannyEdgeDetector.cpp
 * \brief     Canny algorithm class file.
 * \details   This file is part of student project. Some parts of code may be
 *            influenced by various examples found on internet.
 * \author    resset <silentdemon@gmail.com>
 * \date      2006-2012
 * \copyright GNU General Public License, http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt
 */

#include <math.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>
#include <vector>
#include <set>
#include <algorithm>
#include <iterator>

#include "CannyEdgeDetector.hpp"

// --------------------------------------------------------------------------
using namespace cv;
using namespace std;

struct MyCoord {
  int y;
  int x;
};


vector<vector<pair<int, int>>> island_index;
vector<vector<pair<int, int>>> corner_index;
vector<vector<pair<int, int>>> bounding_box_index;
int island_count = 0;

vector<int> island_possibilities;
vector<vector<int>> corner_possibilities;
vector<vector<pair<int, int>>> corners;

Vec3b black = Vec3b(0, 0, 0);
Vec3b white = Vec3b(255, 255, 255);
float PI = 3.141592;



// -------------------------------------------------------------------------
CannyEdgeDetector::CannyEdgeDetector(int type)
{
    gray_type = type;
  width = (unsigned int)0;
  height = (unsigned int)0;
  x = (unsigned int)0;
  y = (unsigned int)0;
  mask_halfsize = (unsigned int)0;
}

CannyEdgeDetector::~CannyEdgeDetector()
{
  delete[] edge_magnitude;
  delete[] edge_direction;
  delete[] workspace_bitmap;
}

uint8_t* CannyEdgeDetector::ProcessImage(uint8_t* source_bitmap, unsigned int width,
  unsigned int height, float sigma,
  uint8_t lowThreshold, uint8_t highThreshold)
{
  /*
   * Setting up image width and height in pixels.
   */
  this->width = width;
  this->height = height;

  /*
   * We store image in array of bytes (chars) in BGR(BGRBGRBGR...) order.
   * Size of the table is width * height * 3 bytes.
   */
  this->source_bitmap = source_bitmap;

  /*
   * Conversion to grayscale. Only luminance information remains.
   */
    switch (gray_type)
    {
        case 1: case 2: case 3:
            this->rgb(gray_type);
            break;
        case 0:
        default:
            this->Luminance();
            break;
    }
  

  /*
   * "Widening" image. At this step we already need to know the size of
   * gaussian mask.
   */
  this->PreProcessImage(sigma);

  /*Mat widening(height, width, CV_8UC3, source_bitmap);
  imshow("after_widening", widening);*/

  /*
   * Noise reduction - Gaussian filter.
   */
  this->GaussianBlur(sigma);
  
  /*Mat gaussian(height, width, CV_8UC3, workspace_bitmap);
  imshow("after_GaussianBlur", gaussian);*/

  /*
   * Edge detection - Sobel filter.
   */
  this->EdgeDetection();

  /*Mat sobel(height, width, CV_8UC3, edge_magnitude);
  imshow("after_sobel_filter", sobel);*/

  /*
   * Suppression of non maximum pixels.
   */
  this->NonMaxSuppression();

  /*Mat non_max_supp(height, width, CV_8UC3, workspace_bitmap);
  imshow("after_non_max_supp", non_max_supp);*/

  /*
   * Hysteresis thresholding.
   */
  this->Hysteresis(lowThreshold, highThreshold);

  /*
   * "Shrinking" image.
   */
  this->PostProcessImage();

  return source_bitmap;
}
uint8_t CannyEdgeDetector::GetPixelValue(unsigned int x, unsigned int y)
{
    return (uint8_t) * (workspace_bitmap + (unsigned long)(x * width + y));
}

inline void CannyEdgeDetector::SetPixelValue(unsigned int x, unsigned int y,
    uint8_t value)
{
    workspace_bitmap[(unsigned long)(x * width + y)] = value;
}

void CannyEdgeDetector::PreProcessImage(float sigma)
{
    // Finding mask size with given sigma.
    mask_size = 2 * round(sqrt(-log(0.3) * 2 * sigma * sigma)) + 1;
    mask_halfsize = mask_size / 2;

    // Enlarging workspace bitmap width and height.
    height += mask_halfsize * 2;
    width += mask_halfsize * 2;
    // Working area.
    workspace_bitmap = new uint8_t[height * width];

    // Edge information arrays.
    edge_magnitude = new float[width * height];
    edge_direction = new uint8_t[width * height];

    // Zeroing direction array.
    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {
            edge_direction[x * width + y] = 0;
        }
    }

    cout << "source size : "<< sizeof(source_bitmap)/sizeof(uint8_t) << endl;
    // Copying image data into work area.
    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {
            // Upper left corner.
            if (x < mask_halfsize && y < mask_halfsize) {
                SetPixelValue(x, y, *(source_bitmap));
            }
            // Bottom left corner.
            else if (x >= height - mask_halfsize && y < mask_halfsize) {
                SetPixelValue(x, y, *(source_bitmap + (height - 2 * mask_halfsize - 1) * 3 * (width - 2 * mask_halfsize)));
            }
            // Upper right corner.
            else if (x < mask_halfsize && y >= width - mask_halfsize) {
                SetPixelValue(x, y, *(source_bitmap + 3 * (width - 2 * mask_halfsize - 1)));
            }
            // Bottom right corner.
            else if (x >= height - mask_halfsize && y >= width - mask_halfsize) {
                SetPixelValue(x, y, *(source_bitmap +
                    (height - 2 * mask_halfsize - 1) * 3 * (width - 2 * mask_halfsize) + 3 * (width - 2 * mask_halfsize - 1)));
            }
            // Upper beam.
            else if (x < mask_halfsize) {
                SetPixelValue(x, y, *(source_bitmap + 3 * (y - mask_halfsize)));
            }
            // Bottom beam.
            else if (x >= height - mask_halfsize) {
                SetPixelValue(x, y, *(source_bitmap +
                    (height - 2 * mask_halfsize - 1) * 3 * (width - 2 * mask_halfsize) + 3 * (y - mask_halfsize)));
            }
            // Left beam.
            else if (y < mask_halfsize) {
                SetPixelValue(x, y, *(source_bitmap +
                    (x - mask_halfsize) * 3 * (width - 2 * mask_halfsize)));
            }
            // Right beam.
            else if (y >= width - mask_halfsize) {
                SetPixelValue(x, y, *(source_bitmap +
                    (x - mask_halfsize) * 3 * (width - 2 * mask_halfsize) + 3 * (width - 2 * mask_halfsize - 1)));
            }
            // The rest of the image.
            else {
                SetPixelValue(x, y, *(source_bitmap +
                    (x - mask_halfsize) * 3 * (width - 2 * mask_halfsize) + 3 * (y - mask_halfsize)));
            }
        }
    }
}

void CannyEdgeDetector::PostProcessImage()
{
    // Decreasing width and height.
    unsigned long i;
    height -= 2 * mask_halfsize;
    width -= 2 * mask_halfsize;

    // Shrinking image.
    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {
            i = (unsigned long)(x * 3 * width + 3 * y);
            *(source_bitmap + i) =
                *(source_bitmap + i + 1) =
                *(source_bitmap + i + 2) = workspace_bitmap[(x + mask_halfsize) * (width + 2 * mask_halfsize) + (y + mask_halfsize)];
        }
    }
}

void CannyEdgeDetector::Luminance()
{
    unsigned long i;
    float gray_value, blue_value, green_value, red_value;

    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {

            // Current "B" pixel position in bitmap table (calculated with x and y values).
            i = (unsigned long)(x * 3 * width + 3 * y);

            // The order of bytes is BGR.
            blue_value = *(source_bitmap + i);
            green_value = *(source_bitmap + i + 1);
            red_value = *(source_bitmap + i + 2);

            // Standard equation from RGB to grayscale.
            gray_value = (uint8_t)(0.299 * red_value + 0.587 * green_value + 0.114 * blue_value);

            // Ultimately making picture grayscale.
            *(source_bitmap + i) =
                *(source_bitmap + i + 1) =
                *(source_bitmap + i + 2) = gray_value;
        }
    }
}

void CannyEdgeDetector::rgb(int type) {
    unsigned long i;
    float value[3];

    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {
            i = (unsigned long)(x * 3 * width + 3 * y);

            value[0] = *(source_bitmap + i + 2); // red
            value[1] = *(source_bitmap + i + 1); // green
            value[2] = *(source_bitmap + i + 0); // blue
            *(source_bitmap + i) =
                *(source_bitmap + i + 1) =
                *(source_bitmap + i + 2) = value[type-1];

        }
    }
}

void CannyEdgeDetector::GaussianBlur(float sigma)
{
    // We already calculated mask size in PreProcessImage.
    long signed_mask_halfsize;
    signed_mask_halfsize = this->mask_halfsize;

    float* gaussianMask;
    gaussianMask = new float[mask_size * mask_size];

    for (int i = -signed_mask_halfsize; i <= signed_mask_halfsize; i++) {
        for (int j = -signed_mask_halfsize; j <= signed_mask_halfsize; j++) {
            gaussianMask[(i + signed_mask_halfsize) * mask_size + j + signed_mask_halfsize]
                = (1 / (2 * PI * sigma * sigma)) * exp(-(i * i + j * j) / (2 * sigma * sigma));
        }
    }

    unsigned long i;
    unsigned long i_offset;
    int row_offset;
    int col_offset;
    float new_pixel;

    for (x = signed_mask_halfsize; x < height - signed_mask_halfsize; x++) {
        for (y = signed_mask_halfsize; y < width - signed_mask_halfsize; y++) {
            new_pixel = 0;
            for (row_offset = -signed_mask_halfsize; row_offset <= signed_mask_halfsize; row_offset++) {
                for (col_offset = -signed_mask_halfsize; col_offset <= signed_mask_halfsize; col_offset++) {
                    i_offset = (unsigned long)((x + row_offset) * width + (y + col_offset));
                    new_pixel += (float)((workspace_bitmap[i_offset])) * gaussianMask[(signed_mask_halfsize + row_offset) * mask_size + signed_mask_halfsize + col_offset];
                }
            }
            i = (unsigned long)(x * width + y);
            workspace_bitmap[i] = new_pixel;
        }
    }

    delete[] gaussianMask;
}

void CannyEdgeDetector::EdgeDetection()
{
    // Sobel masks.
    float Gx[9];
    Gx[0] = 1.0; Gx[1] = 0.0; Gx[2] = -1.0;
    Gx[3] = 2.0; Gx[4] = 0.0; Gx[5] = -2.0;
    Gx[6] = 1.0; Gx[7] = 0.0; Gx[8] = -1.0;
    float Gy[9];
    Gy[0] = -1.0; Gy[1] = -2.0; Gy[2] = -1.0;
    Gy[3] = 0.0; Gy[4] = 0.0; Gy[5] = 0.0;
    Gy[6] = 1.0; Gy[7] = 2.0; Gy[8] = 1.0;

    float value_gx, value_gy;

    float max = 0.0;
    float angle = 0.0;

    // Convolution.
    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {
            value_gx = 0.0;
            value_gy = 0.0;

            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    value_gx += Gx[l * 3 + k] * GetPixelValue((x + 1) + (1 - k),
                        (y + 1) + (1 - l));
                    value_gy += Gy[l * 3 + k] * GetPixelValue((x + 1) + (1 - k),
                        (y + 1) + (1 - l));
                }
            }

            edge_magnitude[x * width + y] = sqrt(value_gx * value_gx + value_gy * value_gy) / 4.0;

            // Maximum magnitude.
            max = edge_magnitude[x * width + y] > max ? edge_magnitude[x * width + y] : max;

            // Angle calculation.
            if ((value_gx != 0.0) || (value_gy != 0.0)) {
                angle = atan2(value_gy, value_gx) * 180.0 / PI;
            }
            else {
                angle = 0.0;
            }
            if (((angle > -22.5) && (angle <= 22.5)) ||
                ((angle > 157.5) && (angle <= -157.5))) {
                edge_direction[x * width + y] = 0;
            }
            else if (((angle > 22.5) && (angle <= 67.5)) ||
                ((angle > -157.5) && (angle <= -112.5))) {
                edge_direction[x * width + y] = 45;
            }
            else if (((angle > 67.5) && (angle <= 112.5)) ||
                ((angle > -112.5) && (angle <= -67.5))) {
                edge_direction[x * width + y] = 90;
            }
            else if (((angle > 112.5) && (angle <= 157.5)) ||
                ((angle > -67.5) && (angle <= -22.5))) {
                edge_direction[x * width + y] = 135;
            }
        }
    }

    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {
            edge_magnitude[x * width + y] =
                255.0f * edge_magnitude[x * width + y] / max;
            SetPixelValue(x, y, edge_magnitude[x * width + y]);
        }
    }
}

void CannyEdgeDetector::NonMaxSuppression()
{
    float pixel_1 = 0;
    float pixel_2 = 0;
    float pixel;

    for (x = 1; x < height - 1; x++) {
        for (y = 1; y < width - 1; y++) {
            if (edge_direction[x * width + y] == 0) {
                pixel_1 = edge_magnitude[(x + 1) * width + y];
                pixel_2 = edge_magnitude[(x - 1) * width + y];
            }
            else if (edge_direction[x * width + y] == 45) {
                pixel_1 = edge_magnitude[(x + 1) * width + y - 1];
                pixel_2 = edge_magnitude[(x - 1) * width + y + 1];
            }
            else if (edge_direction[x * width + y] == 90) {
                pixel_1 = edge_magnitude[x * width + y - 1];
                pixel_2 = edge_magnitude[x * width + y + 1];
            }
            else if (edge_direction[x * width + y] == 135) {
                pixel_1 = edge_magnitude[(x + 1) * width + y + 1];
                pixel_2 = edge_magnitude[(x - 1) * width + y - 1];
            }
            pixel = edge_magnitude[x * width + y];
            if ((pixel >= pixel_1) && (pixel >= pixel_2)) {
                SetPixelValue(x, y, pixel);
            }
            else {
                SetPixelValue(x, y, 0);
            }
        }
    }

    bool change = true;
    while (change) {
        change = false;
        for (x = 1; x < height - 1; x++) {
            for (y = 1; y < width - 1; y++) {
                if (GetPixelValue(x, y) == 255) {
                    if (GetPixelValue(x + 1, y) == 128) {
                        change = true;
                        SetPixelValue(x + 1, y, 255);
                    }
                    if (GetPixelValue(x - 1, y) == 128) {
                        change = true;
                        SetPixelValue(x - 1, y, 255);
                    }
                    if (GetPixelValue(x, y + 1) == 128) {
                        change = true;
                        SetPixelValue(x, y + 1, 255);
                    }
                    if (GetPixelValue(x, y - 1) == 128) {
                        change = true;
                        SetPixelValue(x, y - 1, 255);
                    }
                    if (GetPixelValue(x + 1, y + 1) == 128) {
                        change = true;
                        SetPixelValue(x + 1, y + 1, 255);
                    }
                    if (GetPixelValue(x - 1, y - 1) == 128) {
                        change = true;
                        SetPixelValue(x - 1, y - 1, 255);
                    }
                    if (GetPixelValue(x - 1, y + 1) == 128) {
                        change = true;
                        SetPixelValue(x - 1, y + 1, 255);
                    }
                    if (GetPixelValue(x + 1, y - 1) == 128) {
                        change = true;
                        SetPixelValue(x + 1, y - 1, 255);
                    }
                }
            }
        }
        if (change) {
            for (x = height - 2; x > 0; x--) {
                for (y = width - 2; y > 0; y--) {
                    if (GetPixelValue(x, y) == 255) {
                        if (GetPixelValue(x + 1, y) == 128) {
                            change = true;
                            SetPixelValue(x + 1, y, 255);
                        }
                        if (GetPixelValue(x - 1, y) == 128) {
                            change = true;
                            SetPixelValue(x - 1, y, 255);
                        }
                        if (GetPixelValue(x, y + 1) == 128) {
                            change = true;
                            SetPixelValue(x, y + 1, 255);
                        }
                        if (GetPixelValue(x, y - 1) == 128) {
                            change = true;
                            SetPixelValue(x, y - 1, 255);
                        }
                        if (GetPixelValue(x + 1, y + 1) == 128) {
                            change = true;
                            SetPixelValue(x + 1, y + 1, 255);
                        }
                        if (GetPixelValue(x - 1, y - 1) == 128) {
                            change = true;
                            SetPixelValue(x - 1, y - 1, 255);
                        }
                        if (GetPixelValue(x - 1, y + 1) == 128) {
                            change = true;
                            SetPixelValue(x - 1, y + 1, 255);
                        }
                        if (GetPixelValue(x + 1, y - 1) == 128) {
                            change = true;
                            SetPixelValue(x + 1, y - 1, 255);
                        }
                    }
                }
            }
        }
    }

    // Suppression
    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {
            if (GetPixelValue(x, y) == 128) {
                SetPixelValue(x, y, 0);
            }
        }
    }
}

void CannyEdgeDetector::Hysteresis(uint8_t lowThreshold, uint8_t highThreshold)
{
    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {
            if (GetPixelValue(x, y) >= highThreshold) {
                SetPixelValue(x, y, 255);
                this->HysteresisRecursion(x, y, lowThreshold);
            }
        }
    }

    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {
            if (GetPixelValue(x, y) != 255) {
                SetPixelValue(x, y, 0);
            }
        }
    }
}

void CannyEdgeDetector::HysteresisRecursion(long x, long y, uint8_t lowThreshold)
{
    uint8_t value = 0;

    for (long x1 = x - 1; x1 <= x + 1; x1++) {
        for (long y1 = y - 1; y1 <= y + 1; y1++) {
            if ((x1 < height) & (y1 < width) & (x1 >= 0) & (y1 >= 0)
                & (x1 != x) & (y1 != y)) {

                value = GetPixelValue(x1, y1);
                if (value != 255) {
                    if (value >= lowThreshold) {
                        SetPixelValue(x1, y1, 255);
                        this->HysteresisRecursion(x1, y1, lowThreshold);
                    }
                    else {
                        SetPixelValue(x1, y1, 0);
                    }
                }
            }
        }
    }
}

bool is_color(cv::Mat& img, int row, int col) {

    if (img.rows <= row) {
        return false;
    }
    if (img.cols <= col) {
        return false;
    }

    if (img.at<Vec3b>(row, col) != black) {
        return true;
    }
    else {
        return false;
    }
}

void bfs(cv::Mat& img, int i, int j) {
    queue<MyCoord> q;
    int width = img.cols, height = img.rows;;

    if (size(island_index) <= island_count) {
        vector<pair<int, int>> new_island;
        island_index.push_back(new_island);
        vector<pair<int, int>> new_corner;
        corner_index.push_back(new_corner);
    }


    q.push(MyCoord{ i, j });
    island_index[island_count].push_back(make_pair(i, j));
    img.at<Vec3b>(i, j) = black;

    int dx[8] = { 0, 1, 0, -1, 1, 1, -1, -1};
    int dy[8] = { 1, 0, -1, 0, 1, -1, -1, 1 };
    while (!q.empty()) {
        MyCoord curPair = q.front();
        q.pop();

        bool flag = false;
        for (int k = 0; k < 8; k++) {
            int nextX = curPair.x + dx[k];
            int nextY = curPair.y + dy[k];
            if (0 <= nextX && nextX < width && 0 <= nextY && nextY < height) {
                pair<int, int> nextPair = make_pair(nextY, nextX);
                if (is_color(img, nextPair.first, nextPair.second)) {
                    flag = true;
                    q.push(MyCoord{ nextPair.first, nextPair.second });
                    island_index[island_count].push_back(make_pair(nextPair.first, nextPair.second));
                    img.at<Vec3b>(nextPair.first, nextPair.second) = black;
                }
            }
        }
        if (flag == false) {
            corner_index[island_count].push_back(make_pair(curPair.y, curPair.x));
        }
    }
}

void printIslands() {
    for (int i = 0; i < island_index.size(); i++) {
        printf("%d: ", i);
        for (int j = 0; j < island_index[i].size(); j++) {
                  std::cout << island_index.at(i).at(j).first << ' ';
                  std::cout << island_index.at(i).at(j).second << ' ' << '\n';
        }
        printf("Corner : \n");
        for (int j = 0; j < corner_index[i].size(); j++) {
            std::cout << corner_index.at(i).at(j).first << ' ';
            std::cout << corner_index.at(i).at(j).second << ' ' << '\n';
        }
        printf("\n");
    }
}


bool smallPairFirst(std::pair<int, int> p1, std::pair<int, int> p2) {
    return p1.first < p2.first;
}


bool smallPairSecond(std::pair<int, int> p1, std::pair<int, int> p2) {
    return p1.second < p2.second;
}


bool largePairFirst(std::pair<int, int> p1, std::pair<int, int> p2) {
    return p1.first > p2.first;
}


bool largePairSecond(std::pair<int, int> p1, std::pair<int, int> p2) {
    return p1.second > p2.second;
}

void findBoundingBoxes() {
    for (int i = 0; i < island_count; i++) {
        vector<pair<int, int>> bounding_box_init;
        bounding_box_index.push_back(bounding_box_init);

        // min_row
        auto min_row = *min_element(island_index[i].begin(), island_index[i].end(), smallPairFirst);
        int min_row_value = min_row.first;
        //    printf("%d\n", min_row_value);

          // max_row
        auto max_row = *min_element(island_index[i].begin(), island_index[i].end(), largePairFirst);
        int max_row_value = max_row.first;
        //    printf("%d\n", max_row_value);

          // min_col
        auto min_col = *min_element(island_index[i].begin(), island_index[i].end(), smallPairSecond);
        int min_col_value = min_col.second;
        //    printf("%d\n", min_col_value);

          // max_col
        auto max_col = *min_element(island_index[i].begin(), island_index[i].end(), largePairSecond);
        int max_col_value = max_col.second;
        //    printf("%d\n", max_col_value);

          // add to array
        bounding_box_index[i].push_back(make_pair(min_row_value, min_col_value));
        bounding_box_index[i].push_back(make_pair(min_row_value, max_col_value));
        bounding_box_index[i].push_back(make_pair(max_row_value, min_col_value));
        bounding_box_index[i].push_back(make_pair(max_row_value, max_col_value));
    }
    return;
}

void printBoundingBoxes() {
    for (int i = 0; i < bounding_box_index.size(); i++) {
        printf("®¨°” %d: ", i);
        for (int j = 0; j < bounding_box_index[i].size(); j++) {
            std::cout << bounding_box_index[i][j].first << ' ';
            std::cout << bounding_box_index.at(i).at(j).second << ' ' << '\n';

        }
        printf("\n");
    }
}

void saveImage(Mat& mat, int type, string path) {
    Mat cur;
    mat.copyTo(cur);
    for (int x = 0; x < mat.cols; x++) {
        for (int y = 0; y < mat.rows; y++) {
            if (type == 1) {
                cur.at<Vec3b>(y, x)[0] = 0;
                cur.at<Vec3b>(y, x)[1] = 0;
            }
            if (type == 2) {
                cur.at<Vec3b>(y, x)[0] = 0;
                cur.at<Vec3b>(y, x)[2] = 0;
            }
            if (type == 3) {
                cur.at<Vec3b>(y, x)[1] = 0;
                cur.at<Vec3b>(y, x)[2] = 0;
            }
        }
    }
    imwrite(path, cur);
}


void noiseFilter(Mat& mat, int threshold) {
    //for (int i = 0; i < island_index.size(); i++) {
    //    if (threshold > island_index[i].size()) {
    //        /*cout << i << " : " << island_index[i].size() << endl;
    //        for (int j = 0; j < island_index[i].size(); j++) {
    //            std::cout << island_index.at(i).at(j).first << ' ';
    //            std::cout << island_index.at(i).at(j).second << ' ';
    //            Vec3b v = mat.at<Vec3b>(island_index.at(i).at(j).first, island_index.at(i).at(j).second);
    //            printf("%d, %d, %d\n", v[0], v[1], v[2]);
    //        }*/
    //        for (int j = 0; i < island_index[i].size(); j++) {
    //            mat.at<Vec3b>(island_index.at(i).at(j).first, island_index.at(i).at(j).second) = Vec3b(0, 0, 0);
    //        }
    //    }
    //}
}


void blackAndWhiteReverse(cv::Mat& mat){
  for (int r = 0;  r< mat.rows; r++) {
      for (int c = 0; c < mat.cols; c++) {
        if (mat.at<Vec3b>(r, c) == black){
          mat.at<Vec3b>(r, c) = white;
        } else {
          mat.at<Vec3b>(r, c) = black;
        }
    }
  }
  return;
}

int main() {
    printf("hello");
    Mat originalImage = imread("/Users/yejin/Image-to-Coloring-Book/mac_xcode/coloring_book_generation/coloring_book_generation/puang_pic.jpg");
    int row = originalImage.rows;
    int col = originalImage.cols;
    int imageSize = row * col * originalImage.channels();
    imshow("original", originalImage);

    cout << "Image Size : " << imageSize << endl;

    uint8_t* source_bitmap[4];
    for (int i = 0; i < 4; i++) {
        source_bitmap[i] = new uint8_t[imageSize];
        memcpy(source_bitmap[i], originalImage.data, imageSize);
    }

    CannyEdgeDetector* canny[4];
    for (int i = 0; i < 4; i++) {
        canny[i] = new CannyEdgeDetector(i);
    }
    uint8_t* output[4];
    for (int i = 0; i < 4; i++) {
        output[i] = canny[i]->ProcessImage(source_bitmap[i], col, row, 1.0F, 1, 10);
    }
   
    Mat after_canny[4];
    for (int i = 0; i < 4; i++) {
        after_canny[i] = Mat(row, col, CV_8UC3, output[i]);
    }
    imshow("res0", after_canny[0]);
    imshow("res1", after_canny[1]);
    imshow("res2", after_canny[2]);
    imshow("res3", after_canny[3]);

    // ¿˙¿Â
    saveImage(after_canny[0], 0, "result0.png");
    saveImage(after_canny[1], 1, "result1.png");
    saveImage(after_canny[2], 2, "result2.png");
    saveImage(after_canny[3], 3, "result3.png");

    // R G B µ˚∑Œ«—∞≈ «’ƒ°±‚
    uint8_t* sum_output = new uint8_t[imageSize];
    memcpy(sum_output, output[0], imageSize);
    for (int x = 0; x < col; x++) {
        for (int y = 0; y < row; y++) {
            int index = 3*(x * row + y);
            sum_output[index] = 0;
            sum_output[index+1] = 0;
            sum_output[index+2] = 0;
        }
    }
    for (int x = 0; x < col; x++) {
        for (int y = 0; y < row; y++) {
            int index = 3 * (x * row + y);
            if (output[1][index] || output[2][index] || output[3][index]) {
                sum_output[index] = 255;
                sum_output[index + 1] = 255;
                sum_output[index + 2] = 255;
            }
            else {
                sum_output[index] = 0;
                sum_output[index + 1] = 0;
                sum_output[index + 2] = 0;
            }
        }
    }

    Mat sumMat(row, col, CV_8UC3, sum_output);
    blackAndWhiteReverse(sumMat);
    imshow("sum", sumMat);

    saveImage(sumMat, 0, "result_sum.png");
    cout << "here";
    /*CannyEdgeDetector* canny = new CannyEdgeDetector(0);
    uint8_t* output = canny->ProcessImage(source_bitmap, col, row, 1.0F, 1, 10);*/

   /* Mat after_canny(row, col, CV_8UC3, output);
    imshow("after_canny0", after_canny);

    cv::Mat img = after_canny.clone();*/

   Mat cloneImg = sumMat.clone();
   for (int row = 0; row < cloneImg.rows; row++)
    {
        for (int col = 0; col < cloneImg.cols; col++)
        {
            if (is_color(cloneImg, row, col)) {
                bfs(cloneImg, row, col);
                island_count += 1;
            }
        }
    }
    printf("island_count: %d\n", island_count);
    printIslands();

    /*noiseFilter(sumMat, 15);
    saveImage(sumMat, 0, "result_sum_noise.png");*/
       
   //imshow("rgb", after_canny);
    waitKey(0);

    printf("≥°!");
    return 0;
}
