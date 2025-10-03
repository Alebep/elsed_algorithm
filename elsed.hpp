#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace elsed {

    struct Segment {
        float x1, y1, x2, y2;
    };

    struct Params {
        int    gaussian_ksize = 5;
        double gaussian_sigma = 1.0;
        int    sobel_ksize = 3;

        double grad_thresh = 30.0;   
        double anchor_thresh_frac = 0.2;   

        int    anchor_scan_step = 2;

        int    min_fit_points = 8;
        double max_perp_err = 1.5;    // (px)
        int    max_outliers_in_run = 5;
        double min_length = 10.0;   // (px)

        bool enable_jumps = true;
        std::vector<int> jump_lengths{ 5, 7, 9, 11 };
        int  jump_min_forward_pixels = 3;

        int    st_window_radius = 1;
        double st_ev_ratio_min = 4.0;
        double max_angle_err_deg = 15.0;

        int    val_ignore_end_pixels = 2;
        double val_required_fraction = 0.5;

        
        bool   use_fast_atan2 = true; 
    };

    class ELSED {
    public:
        explicit ELSED(const Params& p = Params()) : params_(p) {}

        
        void detect(const cv::Mat& gray,
            std::vector<Segment>& segments,
            std::vector<float>& scores) const;

        cv::Mat drawSegments(const cv::Mat& img,
            const std::vector<Segment>& segments,
            const cv::Scalar& color = cv::Scalar(0, 255, 0)) const;

    private:
        Params params_;
    };

} 
