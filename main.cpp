#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

#include <emmintrin.h>
#include <immintrin.h>

//#define DPOINT (768*340 + 660)
#define DPOINT (-1)
#define DFRAME (-1)

struct Model {
    int mode;

    int size;
    int frames;

    float* weights0 {};
    float* means0 {};
    float* variances0 {};

    duration<long long int, milli> total {};

    static constexpr float BACKGROUND {0.f};
    static constexpr float FOREGROUND {1.f};

    Model(int mode, cv::Mat& input) : mode{mode}, frames{} {
        size = input.rows * input.cols;
    }

    virtual void run(cv::Mat& input0, cv::Mat& output0) = 0;

    void stats() const {
        stringstream sst;
        sst << "Mode: " << mode << " Frames: " << frames << " Avg: " << (double)total.count() / frames << " millis\n";
        cout << sst.str();
    }
};

struct ModelGM : public Model {
    static constexpr float DEFAULT_VARIANCE { 36.f };
    static constexpr float MIN_VARIANCE { 4.f };
    static constexpr float MAX_VARIANCE { 5 * DEFAULT_VARIANCE };

    static constexpr float THRESHOLD_BACKGROUND { 13.f };
    static constexpr float ALPHA { 0.02f }; //

    ModelGM(int mode, cv::Mat& input) : Model(mode, input) {
        means0 = (float*)aligned_alloc(16, size*sizeof(float));
        variances0 = (float*)aligned_alloc(16, size*sizeof(float));
    }

    void run(cv::Mat& input0, cv::Mat& output0) override {
        ++frames;

        auto alpha {ModelGM::ALPHA};

        auto input = input0.ptr<float>();
        auto output = output0.ptr<float>();

        auto means = means0;
        auto variances = variances0;

        duration<long long int, milli> frame{};

        if(DFRAME > 0 && frames == DFRAME) {
        }

        int stride;
        switch(mode) {
            case 2: stride = 1; break;  // => computeSSE1
            case 3: stride = 4; break;  // => computeSSE4
            case 4: stride = 8; break;  // => computeSSE8
            case 5: stride = 16; break;  // => computeSSE16
            default: mode=1; stride=1; break; // => compute
        }

        auto start = high_resolution_clock::now();

        if(frames > 1) {
            for (int p = 0; p < size; p += stride,
                    input += stride, output += stride, means += stride, variances += stride) {

                if (DPOINT >= 0 && p == DPOINT) {
                }

                switch (mode) {
                    case 1: compute(alpha, input, means, variances, output); break;
                    case 2: computeSSE1(alpha, input, means, variances, output); break;
                    case 3: computeSSE4(alpha, input, means, variances, output); break;
                    case 4: computeSSE8(alpha, input, means, variances, output); break;
                    //case 5: computeSSE16(alpha, input, means, variances, output); break;
                }

                if (DPOINT >= 0 && p == DPOINT) {
                    stringstream ss;
                    ss << "Mode: " << mode << " Frame: " << frames << " Alpha" << alpha;
                    ss << " Px: " << (int) *input << " ";
                    ss << "[M: " << *means << " V: " << *variances << "]" << "\n";
                    cout << ss.str();
                }
            }
        } else {
            for (int p = 0; p < size; p += stride,
                    input += stride, output += stride, means += stride, variances += stride) {
                switch (mode) {
                    case 1: init(input, means, variances, output); break;
                    case 2: initSSE1(input, means, variances, output); break;
                    case 3: initSSE4(input, means, variances, output); break;
                    case 4: initSSE8(input, means, variances, output); break;
                    //case 5: initSSE16(input, means, variances, output); break;
                }
            }
        }

        auto stop = high_resolution_clock::now();
        frame += duration_cast<milliseconds>(stop - start);

        total += frame;
    }

    static inline void init(const float* inputs, float* means, float* variances, float* outputs) {
        *means = *inputs;
        *variances = ModelGM::DEFAULT_VARIANCE;
        *outputs = ModelGM::BACKGROUND;
    }

    static inline void compute(float alpha, const float* inputs, float* means, float* variances, float* outputs) {
        auto input {*inputs};
        auto mean {*means};

        auto d {mean - input};
        auto d2 {d*d};

        auto variance = *variances;

        //decide if the new sample matches the model: (mean - input)^2 <= Threshold * variance
        auto mask {d2 <= ModelGM::THRESHOLD_BACKGROUND * variance};
        auto output{mask ? ModelGM::BACKGROUND : ModelGM::FOREGROUND};

        //update the model accordingly with the new sample
        *means = mean - alpha * d; //fnmadd
        *variances = max(ModelGM::MIN_VARIANCE, min(ModelGM::MAX_VARIANCE, variance + alpha * (d2 - variance))); //fmadd & cap min + max

        *outputs = output;
    }

    static inline void initSSE1(const float* inputs, float* means, float* variances, float* outputs) {
        _mm_store_ss(means, _mm_load_ss(inputs));
        _mm_store_ss(variances, _mm_set_ss(ModelGM::DEFAULT_VARIANCE));
        _mm_store_ss(outputs, _mm_setzero_ps());//ModelGM::BACKGROUND
    }

    static inline void computeSSE1(float alpha, const float* inputs, float* means, float* variances, float* outputs) {
        __m128 input = _mm_load_ss(inputs);
        __m128 mean = _mm_load_ss(means);

        __m128 d = _mm_sub_ss(mean, input);
        __m128 d2 = _mm_mul_ss(d, d);

        __m128 variance = _mm_load_ss(variances);

        __m128 mask = _mm_cmp_ss(d2, _mm_mul_ss(_mm_set_ss(ModelGM::THRESHOLD_BACKGROUND), variance), _CMP_LE_OS);
        __m128 output = _mm_blendv_ps(_mm_set_ss(ModelGM::FOREGROUND), _mm_setzero_ps(), mask);//ModelGM::BACKGROUND

        __m128 a = _mm_set_ss(alpha);
        //_mm_store_ss(means, _mm_sub_ss(mean, _mm_mul_ss(a, d)));
        _mm_store_ss(means, _mm_fnmadd_ss(a, d, mean)); // -(a*b) + c
        //variance = _mm_add_ss(variance, _mm_mul_ss(a, _mm_sub_ss(d2, variance)));
        variance = _mm_fmadd_ss(a, _mm_sub_ss(d2, variance), variance); // (a*b)+c
        _mm_store_ss(variances, _mm_max_ss(_mm_set_ss(ModelGM::MIN_VARIANCE), _mm_min_ss(_mm_set_ss(ModelGM::MAX_VARIANCE), variance))); // cap min and max

        _mm_store_ss(outputs, output);
    }

    static inline void initSSE4(const float* inputs, float* means, float* variances, float* outputs) {
        _mm_store_ps(means, _mm_load_ps(inputs));
        _mm_store_ps(variances, _mm_set1_ps(ModelGM::DEFAULT_VARIANCE));
        _mm_store_ps(outputs, _mm_setzero_ps());//ModelGM::BACKGROUND
    }

    static inline void computeSSE4(float alpha, const float* inputs, float* means, float* variances, float* outputs) {
        __m128 input = _mm_load_ps(inputs);
        __m128 mean = _mm_load_ps(means);

        __m128 d = _mm_sub_ps(mean, input);
        __m128 d2 = _mm_mul_ps(d, d);

        __m128 variance = _mm_load_ps(variances);

        __m128 mask = _mm_cmp_ps(d2, _mm_mul_ps(_mm_set1_ps(ModelGM::THRESHOLD_BACKGROUND), variance), _CMP_LE_OS);
        __m128 output = _mm_blendv_ps(_mm_set1_ps(ModelGM::FOREGROUND), _mm_setzero_ps(), mask);//ModelGM::BACKGROUND

        __m128 a = _mm_set1_ps(alpha);
        //_mm_store_ps(means, _mm_sub_ps(mean, _mm_mul_ps(a, d)));
        _mm_store_ps(means, _mm_fnmadd_ps(a, d, mean)); // -(a*b) + c
        //variance = _mm_add_ps(variance, _mm_mul_ps(a, _mm_sub_ps(d2, variance)));
        variance = _mm_fmadd_ps(a, _mm_sub_ps(d2, variance), variance); // (a*b)+c
        _mm_store_ps(variances, _mm_max_ps(_mm_set1_ps(ModelGM::MIN_VARIANCE), _mm_min_ps(_mm_set1_ps(ModelGM::MAX_VARIANCE), variance))); // cap min and max

        _mm_store_ps(outputs, output);
    }

    static inline void initSSE8(const float* inputs, float* means, float* variances, float* outputs) {
        _mm256_store_ps(means, _mm256_load_ps(inputs));
        _mm256_store_ps(variances, _mm256_set1_ps(ModelGM::DEFAULT_VARIANCE));
        _mm256_store_ps(outputs, _mm256_setzero_ps());//ModelGM::BACKGROUND
    }

    static inline void computeSSE8(float alpha, const float* inputs, float* means, float* variances, float* outputs) {
        __m256 input = _mm256_load_ps(inputs);
        __m256 mean = _mm256_load_ps(means);

        __m256 d = _mm256_sub_ps(mean, input);
        __m256 d2 = _mm256_mul_ps(d, d);

        __m256 variance = _mm256_load_ps(variances);

        __m256 mask = _mm256_cmp_ps(d2, _mm256_mul_ps(_mm256_set1_ps(ModelGM::THRESHOLD_BACKGROUND), variance), _CMP_LE_OS);
        __m256 output = _mm256_blendv_ps(_mm256_set1_ps(ModelGM::FOREGROUND), _mm256_setzero_ps(), mask);//ModelGM::BACKGROUND

        __m256 a = _mm256_set1_ps(alpha);
        //_mm256_store_ps(means, _mm256_sub_ps(mean, _mm256_mul_ps(a, d)));
        _mm256_store_ps(means, _mm256_fnmadd_ps(a, d, mean));
        //variance = _mm256_add_ps(variance, _mm256_mul_ps(a, _mm256_sub_ps(d2, variance)));
        variance = _mm256_fmadd_ps(a, _mm256_sub_ps(d2, variance), variance);
        _mm256_store_ps(variances, _mm256_max_ps(_mm256_set1_ps(ModelGM::MIN_VARIANCE), _mm256_min_ps(_mm256_set1_ps(ModelGM::MAX_VARIANCE), variance)));

        _mm256_store_ps(outputs, output);
    }

    static inline void initSSE16(const float* inputs, float* means, float* variances, float* outputs) {
        _mm512_store_ps(means, _mm512_load_ps(inputs));
        _mm512_store_ps(variances, _mm512_set1_ps(ModelGM::DEFAULT_VARIANCE));
        _mm512_store_ps(outputs, _mm512_setzero_ps());//ModelGM::BACKGROUND
    }

    static inline void computeSSE16(float alpha, const float* inputs, float* means, float* variances, float* outputs) {
        __m512 input = _mm512_load_ps(inputs);
        __m512 mean = _mm512_load_ps(means);

        __m512 d = _mm512_sub_ps(mean, input);
        __m512 d2 = _mm512_mul_ps(d, d);

        __m512 variance = _mm512_load_ps(variances);

        __mmask16 mask = _mm512_cmp_ps_mask(d2, _mm512_mul_ps(_mm512_set1_ps(ModelGM::THRESHOLD_BACKGROUND), variance), _CMP_LE_OS);
        __m512 output = _mm512_mask_blend_ps(mask, _mm512_set1_ps(ModelGM::FOREGROUND), _mm512_setzero_ps());//ModelGM::BACKGROUND

        __m512 a = _mm512_set1_ps(alpha);
        //_mm512_store_ps(means, _mm512_sub_ps(mean, _mm512_mul_ps(a, d)));
        _mm512_store_ps(means, _mm512_fnmadd_ps(a, d, mean));
        //variance = _mm512_add_ps(variance, _mm512_mul_ps(a, _mm512_sub_ps(d2, variance)));
        variance = _mm512_fmadd_ps(a, _mm512_sub_ps(d2, variance), variance);
        _mm512_store_ps(variances, _mm512_max_ps(_mm512_set1_ps(ModelGM::MIN_VARIANCE), _mm512_min_ps(_mm512_set1_ps(ModelGM::MAX_VARIANCE), variance)));

        _mm512_store_ps(outputs, output);
    }
};

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        cout << "Position ( x: " << x << ", y: " << y << " )" << endl;
    }
}

int main(int argc, char** argv ) {
    if ( argc != 4 ) {
        cout << "Usage: gmm <video> <mode> <ref>" << endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);

    auto mode {stoi(argv[2])};
    auto ref {stoi(argv[3])};

    if(!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    cv::Mat rgb, input, inputF;

    // Capture first frame
    cap >> rgb;

    // If the frame is empty, break immediately
    if(!rgb.empty()) {
        cv::Mat outputGMMRef{rgb.rows, rgb.cols, CV_32F, cv::Scalar(0.f)};
        cv::Mat outputGMMTest{rgb.rows, rgb.cols, CV_32F, cv::Scalar(0.f)};

        cout << "Frame: " << rgb.size() << endl;
        cout << "Mode: " << mode << endl;
        cout << "Ref: " << ref << endl;

        cvtColor(rgb, input, cv::COLOR_BGR2GRAY);
        input.convertTo(inputF, CV_32F);

        ModelGM modelGMMRef{1, inputF};
        ModelGM modelGMMTest{mode, inputF};

        if (ref) {
            modelGMMRef.run(inputF, outputGMMRef);
        }
        modelGMMTest.run(inputF, outputGMMTest);

        //namedWindow("Diff");
        //setMouseCallback("Diff", CallBackFunc);

        while (true) {
            // Capture frame-by-frame
            cap >> rgb;
            // If the frame is empty, break immediately
            if (rgb.empty()) break;

            // Convert it into the input for the algorithm
            cvtColor(rgb, input, cv::COLOR_BGR2GRAY);
            input.convertTo(inputF, CV_32F);

            // Run the model
            if (ref) {
                modelGMMRef.run(inputF, outputGMMRef);
            }
            modelGMMTest.run(inputF, outputGMMTest);

            // Display the resulting frames
            imshow("GMM", rgb);
            imshow("GMMTest", outputGMMTest);

            if (ref) {
                imshow("GMMRef", outputGMMRef);
                imshow("Diff", outputGMMRef - outputGMMTest);
            }

            // Press  ESC on keyboard to exit
            if ((char) cv::waitKey(25) == 27) break;
        };

        if (ref) {
            modelGMMRef.stats();
        }
        modelGMMTest.stats();
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}