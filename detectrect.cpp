#include "detectrect.hpp"

namespace detrect {

    cv::Mat findRect(const cv::Mat& image, int th1, int th2, int filter_size) {
        cv::Mat /*gray_image,*/ blurred_image, edges, mask;

        /*if (image.channels() == 3)
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        else
            gray_image = image;*/


        cv::GaussianBlur(image, blurred_image, cv::Size(5, 5), 0);
        cv::Canny(blurred_image, edges, th1, th2, filter_size, true);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        mask = cv::Mat::zeros(image.size(), CV_8UC1);

        // Filtrar contornos verticais
        for (const auto& contour : contours) {
            cv::Rect bounding_rect = cv::boundingRect(contour);
            if (bounding_rect.height > bounding_rect.width * 2.0 && bounding_rect.height > 10) {
                cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, -1, 255, cv::FILLED);
            }
        }

        // Aplicar opera��es morfol�gicas para conectar linhas verticais
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 9));
        cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 1);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::erode(mask, mask, kernel, cv::Point(-1, -1), 1);

        return mask;
    }

    cv::Mat vincosOnImg(cv::Mat& img, int th1, int th2, int filter_size, int line_thickness, int pixel_threshold, bool resize) {
        cv::Mat image;

        if (resize)
            cv::resize(img, image, cv::Size(), 0.25, 0.25);
        else
            image = img;

        cv::Mat mask = findRect(image, th1, th2, filter_size);

        cv::Mat new_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
        int height = mask.rows, width = mask.cols;

        // Calcular soma cumulativa das colunas para otimizar a soma de pixels brancos
        cv::Mat column_sums;
        cv::reduce(mask == 255, column_sums, 0, cv::REDUCE_SUM, CV_32SC1);
        column_sums /= 255; // Converter soma de 255 para contagem de pixels brancos

        std::vector<int> cumulative_sums(width + 1, 0);
        for (int col = 0; col < width; ++col) {
            cumulative_sums[col + 1] = cumulative_sums[col] + column_sums.at<int>(0, col);
        }

        // Processar colunas em paralelo
        std::vector<bool> column_flags(width - line_thickness + 1);

        cv::parallel_for_(cv::Range(0, width - line_thickness + 1), [&](const cv::Range& range) {
            for (int col = range.start; col < range.end; ++col) {
                int white_pixel_sum = cumulative_sums[col + line_thickness] - cumulative_sums[col];
                column_flags[col] = (white_pixel_sum >= pixel_threshold);
            }
            });

        // Atualizar new_mask
        for (int col = 0; col < width - line_thickness + 1; ++col) {
            if (column_flags[col]) {
                new_mask.colRange(col, col + line_thickness).setTo(255);
            }
        }

        // Aplicar operações morfológicas para eliminar protuberâncias e conectar linhas
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
        cv::Mat closed_mask, dilated_mask, final_mask;

        cv::morphologyEx(new_mask, closed_mask, cv::MORPH_CLOSE, kernel);
        cv::dilate(closed_mask, dilated_mask, kernel, cv::Point(-1, -1), 2);
        cv::erode(dilated_mask, final_mask, kernel, cv::Point(-1, -1), 2);
        cv::dilate(final_mask, final_mask, kernel, cv::Point(-1, -1), 4);

        if (resize)
            cv::resize(final_mask, final_mask, cv::Size(img.cols, img.rows));

        return final_mask;
    }

    cv::Mat positiveTonegative(cv::Mat mask) {
        cv::Mat invertMask;
        cv::bitwise_not(mask, invertMask);
        return invertMask;
    }

    void vlinesLimitOnMask(cv::InputArray imagem, std::vector<int>& posicoes_inicio, std::vector<int>& posicoes_fim) {
        cv::Mat binaria;
        threshold(imagem, binaria, 127, 255, cv::THRESH_BINARY);

        int altura = binaria.rows;
        int largura = binaria.cols;

        // Soma dos pixels em cada coluna
        cv::Mat sum_cols;
        reduce(binaria, sum_cols, 0, cv::REDUCE_SUM, CV_32S);

        int* sum_cols_data = sum_cols.ptr<int>(0);
        double threshold_value = altura * 255 * 0.95;

        std::vector<uchar> presenca_linha(largura);

        // vale a pena, mas deve se ter cuidado com as dependencias.
        uchar prev_value = 0;
        for (int i = 0; i < largura; i++) {
            uchar current_value = sum_cols_data[i] > threshold_value ? 1 : 0;
            presenca_linha[i] = current_value;

            // Detectar transições e armazenar posições
            if (i > 0) {
                int transition = current_value - prev_value;
                if (transition == 1) {
                    posicoes_inicio.push_back(i);
                }
                else if (transition == -1) {
                    posicoes_fim.push_back(i);
                }
            }
            prev_value = current_value;
        }

        // Ajustar casos onde a imagem começa ou termina com uma linha
        if (presenca_linha[0]) {
            posicoes_inicio.insert(posicoes_inicio.begin(), 0);
        }
        if (presenca_linha[largura - 1]) {
            posicoes_fim.push_back(largura - 1);
        }
    }

    void apply_masks_not_vcreases(cv::InputArray rect, cv::InputOutputArray detMask, int th1, int th2, int filter_size, int line_thickness, int pixel_threshold, bool resize, int large_vert_inter) {

        int kernel_len = (filter_size == 2) ? 5 : (filter_size == 3) ? 7 : 3;
        cv::Mat rectMask;
        std::vector<int> start_pos, end_pos;

        cv::rotate(rect, rectMask, cv::ROTATE_90_CLOCKWISE);
        cv::rotate(detMask, detMask, cv::ROTATE_90_CLOCKWISE);

        int cols = rectMask.cols, rows = rectMask.rows;

        if (resize)
            cv::resize(rectMask, rectMask, cv::Size(), 0.25, 0.25);

        rectMask = vincosOnImg(rectMask, th1, th2, kernel_len, line_thickness, pixel_threshold);

        vlinesLimitOnMask(rectMask, start_pos, end_pos);

        if (end_pos.size() > 1) {
            // nao vale a pena paralelizar
            size_t max_index = end_pos.size() - 1;
            for (size_t x = 0; x < max_index; x++) {
                if (start_pos[x + 1] - end_pos[x] < large_vert_inter) {
                    int start_col = end_pos[x];
                    int end_col = start_pos[x + 1];
                    int width = end_col - start_col;

                    // Verificar se os valores estão dentro dos limites da imagem
                    if (width > 0 && start_col >= 0 && end_col <= rectMask.cols) {
                        // Definir a região de interesse (ROI) e definir os pixels como zero
                        cv::Rect rect(start_col, 0, width, rectMask.rows);
                        rectMask(rect).setTo(cv::Scalar(255));
                    }
                }
            }
        }

        if (resize)
            cv::resize(rectMask, rectMask, cv::Size(cols, rows));
    }
}