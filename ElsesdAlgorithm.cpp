#include "elsed.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    cv::Mat gray = cv::imread("D:\\Trabalho\\CV\\workdir\\AutoEuropa\\vision1\\2025_capture\\VISION1\\holes\\1407257845-62718301\\original_289.jpg", cv::IMREAD_GRAYSCALE);
    if (gray.empty()) { std::cerr << "Falha ao abrir imagem\n"; return 1; }

    elsed::Params params;
    elsed::ELSED detector(params);

    std::vector<elsed::Segment> segs;
    std::vector<float> scores;

    auto start = std::chrono::high_resolution_clock::now();
    detector.detect(gray, segs, scores);
    auto end = std::chrono::high_resolution_clock::now();

    auto detect_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Segmentos: " << segs.size() << "\n";
    std::cout << "Tempo de deteção: " << detect_duration << " ms\n";
    cv::Mat vis = detector.drawSegments(gray, segs);
    cv::imwrite("elsed_out.png", vis);
    return 0;
}
