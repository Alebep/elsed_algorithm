#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
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
    case 1:
        gray = img_bgr.clone();
        break;
    case 3:
        cv::cvtColor(img_bgr, gray, cv::COLOR_BGR2GRAY);
        break;
    case 4:
        cv::cvtColor(img_bgr, gray, cv::COLOR_BGRA2GRAY);
        break;
    default:
        std::cerr << "Número de canais não suportado: " << img_bgr.channels() << "\n";
        return 1;
    }

    if (gray.type() != CV_8UC1) {
        gray.convertTo(gray, CV_8UC1);
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat creaseMask = detrect::vincosOnImg(gray);
    const auto t1 = std::chrono::high_resolution_clock::now();
    /*/detrect::apply_masks_not_vcreases(
        gray,
        nonCreaseMask,
        30,   // canny_threshold1
        70,   // canny_threshold2
        1,    // vertical_dect_kernel
        6,    // large_col_pixel_dect
        65,   // threshold_vertival_rect_decte
        true, // canny_resize
        35    // large_lines_intervale
    );*/

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[detectrect] Tempo de detecção: " << ms << " ms\n";
    std::cout << "[detectrect] Pixels detectados: " << cv::countNonZero(creaseMask) << "\n";

    cv::Mat result = img_bgr.clone();
    const cv::Scalar red(0, 0, 255);
    result.setTo(red, creaseMask);

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
    std::string image_path = "D:\\Trabalho\\CV\\workdir\\AutoEuropa\\vision1\\2025_capture\\VISION2\\HOLES\\0905252065-61760172\\original_364.jpg";

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
