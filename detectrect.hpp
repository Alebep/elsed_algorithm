#ifndef Rect_Detect
#define Rect_Detect

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

namespace detrect {
	/**
	* @brief Fun��o para encontrar contornoes ret�ngulares cuja a altura � maior que a largura.
	*
	* Esta fun��o recebe dois par�metros uma matriz e um inteiro .
	*
	* @param image uma imagem CV_8UC1.
	* @param th1 é o threshold do canny que define o valor minimo para os pixels serem considerados validos
	* @param th2 é o threshold do canny que define o valor maximo para os pixels serem considerados validos
	* @param filter_size um inteiro que define o tamnaho do filtro de sobel a ser usado no Canny.
	* @return Mat � uma matriz com as linhas retas detedas na imagem.
	*/
	cv::Mat findRect(const cv::Mat& image, int th1 = 30, int th2 = 70, int filter_size = 3);

	/**
	 * @brief Fun��o para detectar vincos retos na imagem e gerar uma mascara com a regi�o com os vincos ativa.
	 *
	 * Esta fun��o pega nas retas geradas pelo findRect e as torna mais perfeitas isso porque as retas geradas pelo findRect
	 * podem estar segmentadas, o que est� fun��o faz � unifica-las para que se tornem retas continuas e com uma espessura que
	 * cubra os vincos.
	 *
	 * @param image uma imagem CV_8UC1.
	 * @param th1 é o limiar minimo do findrect
	 * @param th2 é o limiar maximo do findrect
	 * @param filter_size é o tamnaho do filtro de sobel a ser usado no findrect como parametro do Canny.
	 * @param line_thickness largura(coordenada x) da da reta que procura os pixels ativos, para tra�ar uma reta vertical
	 * que corte a imagem.
	 * @param pixel_threshold limiar de pixels positivos que a regi�o correspondente ao line_thickness e altura da imagem devem ter
	 * para a reta ser desenhada.
	 * @param resize flag que define se a imagem de entrada será redimensionada
	 * @return Mat uma matriz com as retas solidas que cortam a imagem.
	 */
	cv::Mat vincosOnImg(cv::Mat& img, int th1 = 30, int th2 = 70, int filter_size = 3, int line_thickness = 3, int pixel_threshold = 64, bool resize = false);

	/*  @brief Essa fun��o tem como base uma mascara interna com o mesmo tamanho que a mascara de entrada
	 *  e torna todos os pontos positivos da mascara de entrada negativos na mascara interna.
	 *
	 * @param Mat mascara com as retas solidas
	 * @return Mat uma mascara cujo os valores de entrada s�o invertidos.
	 */
	cv::Mat positiveTonegative(cv::Mat mask);

	/*
	* @brief Essa função é a função onde todo o processo para deteção dos vincos acontece
	*
	* Essa função ela é adaptada ao contexto do problema da auto-europa ela não é generica
	*
	* @param rect imagem em que serao feitas as transformacaoes para detecao dos vincos
	* @param detMask imagem que será uma intersecção entre mascara de defeitos original e o inverso dos vincos
	* @param th1 é o limiar minimo do findrect
	* @param th2 é o limiar maximo do findrect
	* @param filter_size é o tamnaho do filtro de sobel a ser usado no findrect como parametro do Canny.
	* @param line_thickness largura(coordenada x) da da reta que procura os pixels ativos, para tra�ar uma reta vertical
	* que corte a imagem.
	* @param pixel_threshold limiar de pixels positivos que a regi�o correspondente ao line_thickness e altura da imagem devem ter
	* para a reta ser desenhada.
	* @param resize flag que define se a imagem de entrada será redimensionada
	* @param large_vert_inter largura maxima( eixo x) do rectangulo que representa um vinco completo
	*/
	void apply_masks_not_vcreases(cv::InputArray rect, cv::InputOutputArray detMask, int th1, int th2, int filter_size, int line_thickness, int pixel_threshold, bool resize, int large_vert_inter);

	/*
	* @brief Esta função pega as rectas da imagem e deteta a recta que marca o inicio de um vinco e
	* a recta que demarca o fim de um vinco.
	*
	*  O objetivo é demarcar onde começa e onde termina cada vinco.
	*
	* A posicao 0 do posicoes_inicio é o inicio do primeiro vinco e a posicao 0 do posicoes_fim é o fim do primeiro vinco
	* e assim sucessivamente.
	*
	* @param imagem de entrada com varias linhas.
	* @param posicoes_inicio lista com as posicoes iniciais de cada vinco.
	* @param posicoes_fim lista com as posicoes finais de cada vinco
	*/
	void vlinesLimitOnMask(cv::InputArray imagem, std::vector<int>& posicoes_inicio, std::vector<int>& posicoes_fim);
}
#endif
