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
        "  " << argv0 << " --method=elsed --image=<caminho>\n"
        "  " << argv0 << " --method=lsd   --image=<caminho>\n"
        "\n"
        "Opções:\n"
        "  --method=elsed|lsd   Seletor do detector (padrão: elsed)\n"
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

    const auto t0 = std::chrono::high_resolution_clock::now();

    detrect::apply_masks_not_vcreases(img_bgr, detection_masked,
        params["canny_threshold1"].asInt(),
        params["canny_threshold2"].asInt(),
        params["vertical_dect_kernel"].asInt(),
        params["large_col_pixel_dect"].asInt(),
        params["threshold_vertival_rect_decte"].asInt(),
        params["canny_resize"].asBool(),
        params["large_lines_intervale"].asInt());    
    
    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[detectrect] Tempo de detecção: " << ms << " ms\n";

    std::vector<std::pair<cv::Point, cv::Point>> lines;
   
    return 0;
}


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Uso: " << argv[0] << " <imagem_input> <mascara_binaria> <saida>\n";
        return 1;
    }

    std::string imgPath = argv[1];
    std::string maskPath = argv[2];
    std::string outPath = argv[3];

    // Lê a imagem (3 canais) e a máscara (1 canal)
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Erro ao carregar imagem: " << imgPath << "\n";
        return 1;
    }
    if (mask.empty()) {
        std::cerr << "Erro ao carregar máscara: " << maskPath << "\n";
        return 1;
    }

    // Garante que a máscara tem o mesmo tamanho da imagem
    if (mask.size() != img.size()) {
        std::cerr << "A máscara e a imagem têm tamanhos diferentes.\n";
        return 1;
    }

    // Se a máscara não estiver estritamente binária, limiariza (opcional, mas útil)
    cv::Mat maskBin;
    cv::threshold(mask, maskBin, 127, 255, cv::THRESH_BINARY);

    // Inverte a máscara
    cv::Mat maskInv;
    cv::bitwise_not(maskBin, maskInv); // 0 <-> 255

    // Cria uma cópia da imagem para editar
    cv::Mat result = img.clone();

    // Define a cor vermelha em BGR (OpenCV usa BGR)
    cv::Vec3b red(0, 0, 255);

    // Aplica vermelho onde maskInv == 255
    // Método 1: usando indexação rápida com setTo e máscara
    result.setTo(red, maskInv);

    // Salva o resultado
    if (!cv::imwrite(outPath, result)) {
        std::cerr << "Erro ao gravar saída em: " << outPath << "\n";
        return 1;
    }

    std::cout << "Feito! Resultado salvo em: " << outPath << "\n";
    return 0;
}


int main(int argc, char** argv)
{
    std::string method = "elsed"; 
    std::string image_path = "D:\\Trabalho\\CV\\workdir\\AutoEuropa\\vision1\\2025_capture\\VISION2\\HOLES\\0905252065-61760172\\original_46.jpg";

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
    else {
        std::cerr << "Método desconhecido: " << method << " (use 'elsed' ou 'lsd')\n";
        print_usage(argv[0]);
        return 2;
    }
}
