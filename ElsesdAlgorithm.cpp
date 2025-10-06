#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <iterator>
#include <cmath>
#include "elsed.hpp"
#include "detetion.hpp"
#include "detectrect.hpp"

static void draw_and_save(
    const cv::Mat& srcBgr,
    const std::vector<std::pair<cv::Point, cv::Point>>& lines,
    const std::string& out_path)
{
    cv::Mat vis = srcBgr.clone();
    for (const auto& ln : lines) {
        cv::line(vis, ln.first, ln.second, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    if (!cv::imwrite(out_path, vis)) {
        std::cerr << "Falha ao salvar: " << out_path << "\n";
    }
    else {
        std::cout << "Resultado salvo em: " << out_path << "\n";
    }
}

static void print_usage(const char* argv0) {
    std::cout <<
        "Uso:\n"
        "  " << argv0 << " --method=elsed      --image=<caminho>\n"
        "  " << argv0 << " --method=lsd        --image=<caminho>\n"
        "  " << argv0 << " --method=detectrect --image=<caminho>\n"
        "\n"
        "Opções:\n"
        "  --method=elsed|lsd   Seletor do detector (padrão: elsed)\n"
        "  --method=elsed|lsd|detectrect   Seletor do detector (padrão: elsed)\n"
        "  --image=<path>       Caminho da imagem de entrada\n";
}

namespace {

    struct DetectrectPreprocessResult {
        cv::Mat rotatedGray;
        cv::Mat detectionMask;
    };

    static void fast_imadjust(const cv::Mat1b& src, cv::Mat1b& dst, int tol = 1, cv::Vec2i in = cv::Vec2i(0, 255), cv::Vec2i out = cv::Vec2i(0, 255)) {

        // src : input CV_8UC1 image
        // dst : output CV_8UC1 imge
        // tol : tolerance, from 0 to 100.
        // in  : src image bounds
        // out : dst image buonds

        tol = std::max(0, std::min(100, tol));

        cv::Mat histMat; float range[] = { 0, 256 };  const float* histRange = { range };  int histSize = 256;
        cv::calcHist(&src, 1, 0, cv::Mat(), histMat, 1, &histSize, &histRange, true, false);
        std::vector<int> histVec(histSize);
        for (int i = 0; i < histSize; ++i) {
            histVec[i] = static_cast<int>(std::round(histMat.at<float>(i)));
        }

        std::vector<int> cum = histVec;
        for (int i = 1; i < static_cast<int>(histVec.size()); ++i) {
            cum[i] = cum[i - 1] + histVec[i];
        }

        // Compute bounds
        int total = src.rows * src.cols;
        int low_bound = total * tol / 100;
        int upp_bound = total * (100 - tol) / 100;
        in[0] = static_cast<int>(std::distance(cum.begin(), std::lower_bound(cum.begin(), cum.end(), low_bound)));
        in[1] = static_cast<int>(std::distance(cum.begin(), std::lower_bound(cum.begin(), cum.end(), upp_bound)));

        // Stretching
        float scale1 = float(out[1] - out[0]) / float(in[1] - in[0]);
        cv::Mat vs = src - in[0];

        vs.setTo(0, src < in[0]);

        cv::Mat vd, vd32, vd32rd;

        vs.convertTo(vd32, CV_32FC1, scale1, 0.5);
        vd32.convertTo(vd, CV_8U);
        vd.convertTo(vd32rd, CV_32FC1);

        bool SLOWER_IMADJUST = true;

        if (SLOWER_IMADJUST) {
            cv::Mat mask_gt0;
            cv::compare(vd32rd - vd32, 0, mask_gt0, cv::CMP_GT);

            cv::Mat mask_gt0_32;
            mask_gt0.convertTo(mask_gt0_32, CV_32F);

            cv::Mat vd32final = vd32rd - (mask_gt0_32 / 255);
            vd32final.convertTo(vd, CV_8U);
        }

        vd.setTo(out[1], vd > out[1]);

        dst = vd;
    }

    static DetectrectPreprocessResult preprocess_for_detectrect(const cv::Mat& gray)
    {
        CV_Assert(!gray.empty());

        cv::Mat workGray;
        if (gray.type() == CV_8UC1)
            workGray = gray.clone();
        else
            gray.convertTo(workGray, CV_8UC1);

        constexpr int kMedianKernel = 5;
        constexpr double kGaussianSigma = 1.0;
        constexpr int kBilateralDiameter = 5;
        constexpr double kBilateralSigmaColor = 22.0;
        constexpr double kBilateralSigmaSpace = 5.0;
        constexpr double kThrGx = 40.0;
        constexpr double kThrGy = 40.0;

        if (workGray.cols >= kMedianKernel && workGray.rows >= kMedianKernel)
            cv::medianBlur(workGray, workGray, kMedianKernel);

        cv::GaussianBlur(workGray, workGray, cv::Size(3, 3), kGaussianSigma);

        cv::Mat I3;
        cv::bilateralFilter(workGray, I3, kBilateralDiameter, kBilateralSigmaColor, kBilateralSigmaSpace);
        workGray = I3;

        cv::Mat1b adjusted;
        fast_imadjust(workGray, adjusted);

        cv::Mat Gx, Gy;
        cv::spatialGradient(adjusted, Gx, Gy, 3, cv::BORDER_REPLICATE);

        cv::Mat Gx32f, Gy32f;
        Gx.convertTo(Gx32f, CV_32F);
        Gy.convertTo(Gy32f, CV_32F);

        cv::medianBlur(Gx32f, Gx32f, kMedianKernel);
        cv::medianBlur(Gy32f, Gy32f, kMedianKernel);

        cv::Mat absGx, absGy;
        cv::absdiff(Gx32f, cv::Scalar::all(0), absGx);
        cv::absdiff(Gy32f, cv::Scalar::all(0), absGy);

        cv::Mat maskGx = absGx > kThrGx;
        cv::Mat maskGy = absGy > kThrGy;
        cv::Mat detection = cv::Mat::zeros(absGx.size(), CV_8U);
        detection.setTo(255, maskGx | maskGy);

        return { workGray, detection };
    }

} 

static int run_elsed(const cv::Mat& img_bgr, const std::string& out_path)
{
    cv::Mat gray;
    if (img_bgr.channels() == 1) gray = img_bgr;
    else cv::cvtColor(img_bgr, gray, cv::COLOR_BGR2GRAY);

    elsed::Params params;
    elsed::ELSED detector(params);

    std::vector<elsed::Segment> segs;
    std::vector<float> scores;

    if (gray.type() != CV_8UC1) {
        gray.convertTo(gray, CV_8UC1);
    }


    const auto t0 = std::chrono::high_resolution_clock::now();
    detector.detect(gray, segs, scores);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[ELSED] Segmentos: " << segs.size() << "\n";
    std::cout << "[ELSED] Tempo de detecção: " << ms << " ms\n";

    cv::Mat vis = detector.drawSegments(img_bgr, segs);

    if (!cv::imwrite(out_path, vis)) {
        std::cerr << "Falha ao salvar: " << out_path << "\n";
        return 1;
    }
    std::cout << "Resultado salvo em: " << out_path << "\n";
    return 0;
}


static int run_lsd(const cv::Mat& img_bgr, const std::string& out_path)
{
    using namespace ls;

    if (img_bgr.empty()) {
        std::cerr << "Imagem vazia.\n";
        return 1;
    }

    auto lsd = createLSDDetector(img_bgr);
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto segments = lsd->detectLines();
    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[LSD] Segmentos: " << segments.size() << "\n";
    std::cout << "[LSD] Tempo de detecção: " << ms << " ms\n";

    std::vector<std::pair<cv::Point, cv::Point>> lines;
    lines.reserve(segments.size());
    for (const auto& s : segments) {
        lines.emplace_back(s.p1, s.p2);
    }
    draw_and_save(img_bgr, lines, out_path);
    return 0;
}

static int run_detectrect(const cv::Mat& img_bgr, const std::string& out_path)
{
    using namespace ls;

    if (img_bgr.empty()) {
        std::cerr << "Imagem vazia.\n";
        return 1;
    }

    cv::Mat gray;
    switch (img_bgr.channels()) {
    case 1:  gray = img_bgr.clone(); break;
    case 3:  cv::cvtColor(img_bgr, gray, cv::COLOR_BGR2GRAY); break;
    case 4:  cv::cvtColor(img_bgr, gray, cv::COLOR_BGRA2GRAY); break;
    default:
        std::cerr << "Número de canais não suportado: " << img_bgr.channels() << "\n";
        return 1;
    }

    if (gray.type() != CV_8UC1) gray.convertTo(gray, CV_8UC1);

    constexpr int  kCannyThreshold1 = 30;
    constexpr int  kCannyThreshold2 = 70;
    constexpr int  kVerticalKernel = 1;
    constexpr int  kLargeColumnPixels = 6;
    constexpr int  kPixelThreshold = 65;
    constexpr bool kResizeForCanny = true;
    constexpr int  kLargeLinesInterval = 35;

    auto prep = preprocess_for_detectrect(gray);

    cv::Mat detectionMask = prep.detectionMask.clone();

    const auto t0 = std::chrono::high_resolution_clock::now();

    detrect::apply_masks_not_vcreases(
        prep.rotatedGray,     // fonte
        detectionMask,        // máscara (será ROTACIONADA pela API!)
        kCannyThreshold1,
        kCannyThreshold2,
        kVerticalKernel,
        kLargeColumnPixels,
        kPixelThreshold,
        kResizeForCanny,
        kLargeLinesInterval
    );

    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // ✅ A API roda a máscara 90° CW; desfazemos aqui (90° CCW) para alinhar com a imagem original
    cv::rotate(detectionMask, detectionMask, cv::ROTATE_90_COUNTERCLOCKWISE);

    // (Opcional) garantir mesmo tamanho da imagem de saída
    if (detectionMask.size() != img_bgr.size()) {
        cv::resize(detectionMask, detectionMask, img_bgr.size(), 0, 0, cv::INTER_NEAREST);
    }

    std::cout << "[detectrect] Tempo de detecção: " << ms << " ms\n";
    std::cout << "[detectrect] Pixels detectados: " << cv::countNonZero(detectionMask) << "\n";

    cv::Mat result = img_bgr.clone();
    result.setTo(cv::Scalar(0, 0, 255), detectionMask);

    if (!cv::imwrite(out_path, result)) {
        std::cerr << "Falha ao salvar: " << out_path << "\n";
        return 1;
    }

    std::cout << "Resultado salvo em: " << out_path << "\n";
    return 0;
}




int main(int argc, char** argv)
{
    std::string method = "detectrect"; 
    std::string image_path = "C:\\Users\\Alebex\\Downloads\\test.jpg";//"D:\\Trabalho\\CV\\workdir\\AutoEuropa\\vision1\\2025_capture\\VISION2\\HOLES\\0905252065-61760172\\original_364.jpg";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        const auto eq = a.find('=');
        if (eq != std::string::npos) {
            const auto key = a.substr(0, eq);
            const auto val = a.substr(eq + 1);
            if (key == "--method") method = val;
            else if (key == "--image") image_path = val;
        }
    }

    if (image_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    cv::Mat img_bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
        std::cerr << "Falha ao abrir imagem: " << image_path << "\n";
        return 1;
    }

    std::string out_path;
    if (method == "elsed") {
        out_path = "elsed_out.png";
        return run_elsed(img_bgr, out_path);
    }
    else if (method == "lsd") {
        out_path = "lsd_out.png";
        return run_lsd(img_bgr, out_path);
    }
    else if (method == "detectrect") {
        out_path = "detectrect_out.png";
        return run_detectrect(img_bgr, out_path);
    }
    else {
        std::cerr << "Método desconhecido: " << method << " (use 'elsed', 'lsd' ou 'detectrect')\n";
        print_usage(argv[0]);
        return 2;
    }
}
