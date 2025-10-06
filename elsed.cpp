#include "elsed.hpp"
#include <deque>
#include <stdexcept>
#include <cmath> 
#include <algorithm>
#include <limits>

namespace elsed {

    namespace {
        constexpr float  EPSF = 1e-9f;
        constexpr double EPS = 1e-9;
        constexpr double kPI = 3.14159265358979323846;

        inline int sgn(float v) { return (v > 0.f) - (v < 0.f); }
        inline int sgn(double v) { return (v > 0.0) - (v < 0.0); }
        inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

        cv::Mat gaussian_blur(const cv::Mat& img, int ksize, double sigma) {
            int k = std::max(3, ksize | 1); 
            cv::Mat out;
            cv::GaussianBlur(img, out, cv::Size(k, k), sigma, sigma, cv::BORDER_REPLICATE);
            return out;
        }

        void sobel_gradients(const cv::Mat& gray, int ksize,
            cv::Mat& gx, cv::Mat& gy, cv::Mat& mag)
        {
            cv::Sobel(gray, gx, CV_32F, 1, 0, ksize);
            cv::Sobel(gray, gy, CV_32F, 0, 1, ksize);
            cv::magnitude(gx, gy, mag);
        }

        cv::Mat quantize_orientation(const cv::Mat& gx, const cv::Mat& gy) {
            cv::Mat q(gx.size(), CV_8U);
            for (int y = 0; y < gx.rows; ++y) {
                const float* px = gx.ptr<float>(y);
                const float* py = gy.ptr<float>(y);
                uint8_t* pq = q.ptr<uint8_t>(y);
                for (int x = 0; x < gx.cols; ++x) {
                    pq[x] = (std::abs(px[x]) < std::abs(py[x])) ? 1u : 0u;
                }
            }
            return q;
        }

        struct OnlineTLS {
            int n = 0;
            double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0;

            void add(double x, double y) {
                ++n; sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y;
            }
            void remove(double x, double y) {
                --n; sx -= x; sy -= y; sxx -= x * x; sxy -= x * y; syy -= y * y;
            }
            int size() const { return n; }

            cv::Point2f mean() const {
                if (n == 0) return cv::Point2f(0.f, 0.f);
                double inv = 1.0 / static_cast<double>(n);
                return cv::Point2f(static_cast<float>(sx * inv), static_cast<float>(sy * inv));
            }

            static void eigen_symmetric_2x2(double a, double b, double c,
                double& l0, double& l1,
                cv::Point2f& v0, cv::Point2f& v1)
            {
                double tr = a + c;
                double det = a * c - b * b;
                double disc = std::sqrt(std::max(0.0, tr * tr - 4.0 * det));
                l0 = 0.5 * (tr - disc);
                l1 = 0.5 * (tr + disc);

                auto eigenvec = [&](double l)->cv::Point2f {
                    double x = b;
                    double y = l - a;
                    if (std::abs(x) + std::abs(y) < 1e-12) {
                        x = l - c;
                        y = b;
                    }
                    float nx = static_cast<float>(x);
                    float ny = static_cast<float>(y);
                    float nrm = std::hypot(nx, ny) + EPSF;
                    return cv::Point2f(nx / nrm, ny / nrm);
                    };
                v0 = eigenvec(l0);
                v1 = eigenvec(l1);
            }

            void eig(cv::Point2f& center, cv::Point2f& dir, cv::Vec2f& lambdas) const {
                center = mean();
                if (n < 2) {
                    dir = cv::Point2f(1.f, 0.f); lambdas = cv::Vec2f(0.f, 0.f); return;
                }
                double sxx_c = sxx - sx * center.x;
                double sxy_c = sxy - sx * center.y;
                double syy_c = syy - sy * center.y;

                double l0, l1; cv::Point2f v0, v1;
                eigen_symmetric_2x2(sxx_c, sxy_c, syy_c, l0, l1, v0, v1);

                if (l1 >= l0) { dir = v1; lambdas = cv::Vec2f(static_cast<float>(l0), static_cast<float>(l1)); }
                else { dir = v0; lambdas = cv::Vec2f(static_cast<float>(l1), static_cast<float>(l0)); }

                float nrm = std::hypot(dir.x, dir.y) + EPSF;
                dir.x /= nrm; dir.y /= nrm;
            }

            double perp_distance_sq(double x, double y) const {
                cv::Point2f c, d; cv::Vec2f ev;
                OnlineTLS* self = const_cast<OnlineTLS*>(this);
                self->eig(c, d, ev);
                float nx = -d.y, ny = d.x;
                float dx = static_cast<float>(x - c.x);
                float dy = static_cast<float>(y - c.y);
                float proj = nx * dx + ny * dy;
                return static_cast<double>(proj * proj);
            }
        };

        struct WalkState { int x, y; cv::Point prev_dir; };

        std::vector<cv::Point> neighbors_by_orientation(const cv::Point& prev, int orientation)
        {
            int dx = prev.x, dy = prev.y;
            std::vector<cv::Point> out;
            if (dx == 0 && dy == 0) {
                if (orientation == 0) { 
                    out = { {-1,0}, {1,0}, {-1,-1}, {1,1}, {-1,1}, {1,-1} };
                }
                else {
                    out = { {0,-1}, {0,1}, {-1,-1}, {1,1}, {-1,1}, {1,-1} };
                }
                return out;
            }
            if (orientation == 0) {
                cv::Point base = (dx < 0) ? cv::Point(-1, 0) : (dx > 0) ? cv::Point(1, 0) : cv::Point(1, 0);
                out = { base, cv::Point(base.x,-1), cv::Point(base.x,1) };
            }
            else {
                cv::Point base = (dy < 0) ? cv::Point(0, -1) : (dy > 0) ? cv::Point(0, 1) : cv::Point(0, 1);
                out = { base, cv::Point(-1,base.y), cv::Point(1,base.y) };
            }
            return out;
        }

        int try_forward_extend(int sx, int sy, const cv::Point2f& d,
            const cv::Mat& mag, const cv::Mat& gx, const cv::Mat& gy,
            const cv::Mat& visited, const Params& params)
        {
            (void)gx; (void)gy; 
            const int H = mag.rows, W = mag.cols;
            int x = sx, y = sy, cnt = 0;
            for (int it = 0; it < 16; ++it) {
                if (x < 0 || x >= W || y < 0 || y >= H) break;
                if (visited.at<uint8_t>(y, x)) break;
                if (mag.at<float>(y, x) < static_cast<float>(params.grad_thresh)) break;
                ++cnt;
                cv::Point step((std::abs(d.x) >= std::abs(d.y)) ? sgn(d.x) : 0,
                    (std::abs(d.y) > std::abs(d.x)) ? sgn(d.y) : 0);
                if (step.x == 0 && step.y == 0) break;
                x += step.x; y += step.y;
            }
            return cnt;
        }

        std::vector<cv::Point> draw_from_anchor(
            int ax, int ay, int /*orient0*/,
            const cv::Mat& gx, const cv::Mat& gy, const cv::Mat& mag,
            cv::Mat& visited, const Params& params)
        {
            const int H = mag.rows, W = mag.cols;
            std::vector<cv::Point> chain;
            std::deque<WalkState> stack;
            stack.push_back({ ax, ay, cv::Point(0,0) });

            while (!stack.empty()) {
                WalkState st = stack.back(); stack.pop_back();
                int x = st.x, y = st.y;
                cv::Point prev_dir = st.prev_dir;

                OnlineTLS tls;
                bool segment_mode = false;

                while (true) {
                    if (x < 0 || x >= W || y < 0 || y >= H) break;
                    if (visited.at<uint8_t>(y, x)) break;
                    if (mag.at<float>(y, x) < static_cast<float>(params.grad_thresh)) break;

                    visited.at<uint8_t>(y, x) = 1;
                    chain.emplace_back(x, y);

                    tls.add(x + 0.5, y + 0.5);
                    if (!segment_mode && tls.size() >= params.min_fit_points) {
                        cv::Point2f c, d; cv::Vec2f ev;
                        tls.eig(c, d, ev);
                        if (ev[1] > 0.f && (ev[1] / (ev[0] + static_cast<float>(EPS)) >= static_cast<float>(params.st_ev_ratio_min))) {
                            segment_mode = true;
                        }
                    }

                    int o_here = (std::abs(gx.at<float>(y, x)) >= std::abs(gy.at<float>(y, x))) ? 0 : 1;
                    int orientation = o_here;
                    if (segment_mode) {
                        cv::Point2f c, d; cv::Vec2f ev; tls.eig(c, d, ev);
                        orientation = (std::abs(d.x) > std::abs(d.y)) ? 0 : 1;
                    }

                    std::vector<cv::Point> candidates = neighbors_by_orientation(prev_dir, orientation);
                    cv::Point best_xy; cv::Point best_step; float best_val = -1.0f; bool found = false;
                    for (size_t i = 0; i < candidates.size(); ++i) {
                        cv::Point cand = candidates[i];
                        int nx = x + cand.x, ny = y + cand.y;
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H && !visited.at<uint8_t>(ny, nx)) {
                            float g = mag.at<float>(ny, nx);
                            if (g >= static_cast<float>(params.grad_thresh) && g > best_val) {
                                int o2 = (std::abs(gx.at<float>(ny, nx)) >= std::abs(gy.at<float>(ny, nx))) ? 0 : 1;
                                if (o2 == orientation) {
                                    best_xy = cv::Point(nx, ny);
                                    best_step = cand;
                                    best_val = g;
                                    found = true;
                                }
                            }
                        }
                    }
                    if (!found) {
                        static const cv::Point diags[4] = { cv::Point(-1,-1),cv::Point(1,1),cv::Point(-1,1),cv::Point(1,-1) };
                        for (int i = 0; i < 4; ++i) {
                            cv::Point cand = diags[i];
                            int nx = x + cand.x, ny = y + cand.y;
                            if (nx >= 0 && nx < W && ny >= 0 && ny < H && !visited.at<uint8_t>(ny, nx)) {
                                float g = mag.at<float>(ny, nx);
                                if (g >= static_cast<float>(params.grad_thresh) && g > best_val) {
                                    best_xy = cv::Point(nx, ny); best_step = cand; best_val = g; found = true;
                                }
                            }
                        }
                    }

                    if (!found) {
                        if (params.enable_jumps && segment_mode) {
                            cv::Point2f c, d; cv::Vec2f ev; tls.eig(c, d, ev);
                            for (size_t i = 0; i < params.jump_lengths.size(); ++i) {
                                int L = params.jump_lengths[i];
                                if (tls.size() < L) continue;
                                int tx = static_cast<int>(std::lround(x + d.x * L));
                                int ty = static_cast<int>(std::lround(y + d.y * L));
                                if (tx < 0 || tx >= W || ty < 0 || ty >= H) continue;
                                if (mag.at<float>(ty, tx) < static_cast<float>(params.grad_thresh)) continue;
                                int fwd = try_forward_extend(tx, ty, d, mag, gx, gy, visited, params);
                                if (fwd >= params.jump_min_forward_pixels) {
                                    cv::Point step((std::abs(d.x) >= std::abs(d.y)) ? sgn(d.x) : 0,
                                        (std::abs(d.y) > std::abs(d.x)) ? sgn(d.y) : 0);
                                    stack.push_back({ tx, ty, step });
                                    break;
                                }
                            }
                        }
                        break; 
                    }

                    prev_dir = best_step;
                    x = best_xy.x;
                    y = best_xy.y;
                }
            }

            return chain;
        }

        std::pair<bool, float> validate_segment(const std::vector<cv::Point>& pts,
            const cv::Point2f& c,
            const cv::Point2f& d,
            const cv::Mat& gx, const cv::Mat& gy,
            const Params& params)
        {
            cv::Point2f nrm(-d.y, d.x);

            std::vector<float> cosang; cosang.reserve(pts.size());
            for (size_t i = 0; i < pts.size(); ++i) {
                int x = clampi(pts[i].x, 0, gx.cols - 1);
                int y = clampi(pts[i].y, 0, gx.rows - 1);
                float gxs = gx.at<float>(y, x), gys = gy.at<float>(y, x);
                float gm = std::hypot(gxs, gys) + EPSF;
                float gxn = gxs / gm, gyn = gys / gm;
                float ca = std::abs(gxn * nrm.x + gyn * nrm.y);
                cosang.push_back(ca);
            }

            int k0 = params.val_ignore_end_pixels;
            if (static_cast<int>(cosang.size()) <= 2 * k0) return std::make_pair(false, 0.0f);

            int beg = k0;
            int end = static_cast<int>(cosang.size()) - k0;

            double thr = std::cos(params.max_angle_err_deg * kPI / 180.0);
            int good = 0, total = end - beg;
            for (int i = beg; i < end; ++i) if (cosang[i] >= static_cast<float>(thr)) ++good;
            float frac = (total > 0) ? (static_cast<float>(good) / static_cast<float>(total)) : 0.0f;

            return std::make_pair(frac >= static_cast<float>(params.val_required_fraction), frac);
        }

        void validate_and_split(const std::vector<cv::Point>& chain,
            const cv::Mat& gx, const cv::Mat& gy,
            const Params& params,
            std::vector<Segment>& segments,
            std::vector<float>& scores)
        {
            if (chain.empty()) return;

            OnlineTLS tls;
            std::vector<cv::Point> buf;
            buf.reserve(chain.size());
            int outliers_run = 0;

            struct Flusher {
                static void flush(OnlineTLS& tls, std::vector<cv::Point>& buf,
                    const Params& params, const cv::Mat& gx, const cv::Mat& gy,
                    std::vector<Segment>& segments, std::vector<float>& scores)
                {
                    if (tls.size() < params.min_fit_points) { tls = OnlineTLS(); buf.clear(); return; }
                    cv::Point2f c, d; cv::Vec2f ev; tls.eig(c, d, ev);
                    float dnrm = std::hypot(d.x, d.y);
                    if (dnrm < EPSF) { tls = OnlineTLS(); buf.clear(); return; }

                    int n = static_cast<int>(buf.size());
                    if (n == 0) { tls = OnlineTLS(); buf.clear(); return; }
                    int i0 = 0, i1 = 0;
                    float vmin = std::numeric_limits<float>::max();
                    float vmax = -std::numeric_limits<float>::max();
                    for (int i = 0; i < n; ++i) {
                        float vx = static_cast<float>(buf[i].x) - c.x;
                        float vy = static_cast<float>(buf[i].y) - c.y;
                        float val = vx * d.x + vy * d.y;
                        if (val < vmin) { vmin = val; i0 = i; }
                        if (val > vmax) { vmax = val; i1 = i; }
                    }
                    cv::Point2f p0(static_cast<float>(buf[i0].x), static_cast<float>(buf[i0].y));
                    cv::Point2f p1(static_cast<float>(buf[i1].x), static_cast<float>(buf[i1].y));
                    float len = std::hypot(p1.x - p0.x, p1.y - p0.y);
                    if (len < static_cast<float>(params.min_length)) { tls = OnlineTLS(); buf.clear(); return; }

                    std::pair<bool, float> vs = validate_segment(buf, c, d, gx, gy, params);
                    bool ok = vs.first; float score = vs.second;
                    if (ok) {
                        Segment s; s.x1 = p0.x; s.y1 = p0.y; s.x2 = p1.x; s.y2 = p1.y;
                        segments.push_back(s);
                        scores.push_back(score);
                    }
                    tls = OnlineTLS(); buf.clear(); return;
                }
            };

            for (size_t idx = 0; idx < chain.size(); ++idx) {
                cv::Point pt = chain[idx];
                tls.add(pt.x + 0.5, pt.y + 0.5);
                buf.push_back(pt);
                if (tls.size() >= params.min_fit_points) {
                    double d2 = tls.perp_distance_sq(pt.x + 0.5, pt.y + 0.5);
                    if (d2 > params.max_perp_err * params.max_perp_err) ++outliers_run;
                    else outliers_run = 0;
                    if (outliers_run >= params.max_outliers_in_run) {
                        for (int k = 0; k < params.max_outliers_in_run; ++k) {
                            cv::Point last = buf.back(); buf.pop_back();
                            tls.remove(last.x + 0.5, last.y + 0.5);
                        }
                        Flusher::flush(tls, buf, params, gx, gy, segments, scores);
                        outliers_run = 0;
                    }
                }
            }
            Flusher::flush(tls, buf, params, gx, gy, segments, scores);
        }

    } 

    void ELSED::detect(const cv::Mat& gray,
        std::vector<Segment>& segments,
        std::vector<float>& scores) const
    {
        if (gray.empty() || gray.channels() != 1) {
            throw std::invalid_argument("ELSED::detect espera imagem 8-bit grayscale (H x W).");
        }
        cv::Mat g8;
        if (gray.type() != CV_8U) {
            gray.convertTo(g8, CV_8U);
        }
        else g8 = gray;

        cv::Mat blur = gaussian_blur(g8, params_.gaussian_ksize, params_.gaussian_sigma);
        cv::Mat gx, gy, mag;
        sobel_gradients(blur, params_.sobel_ksize, gx, gy, mag);

        double gmax; cv::minMaxLoc(mag, nullptr, &gmax);
        double anchor_thr = params_.anchor_thresh_frac * (gmax + EPS);
        double grad_thr = std::max(params_.grad_thresh, 0.05 * (gmax + EPS));

        cv::Mat qori = quantize_orientation(gx, gy);

        const int H = mag.rows, W = mag.cols;
        std::vector<cv::Point> anchors;
        int step = std::max(1, params_.anchor_scan_step);

        for (int y = 0; y < H; y += step) {
            for (int x = 0; x < W; x += step) {
                float m = mag.at<float>(y, x);
                if (m < static_cast<float>(std::max(anchor_thr, grad_thr))) continue;
                bool ok = false;
                if (qori.at<uint8_t>(y, x) == 0) { 
                    if (x > 0 && x < W - 1) ok = (mag.at<float>(y, x) >= mag.at<float>(y, x - 1) &&
                        mag.at<float>(y, x) > mag.at<float>(y, x + 1));
                }
                else { 
                    if (y > 0 && y < H - 1) ok = (mag.at<float>(y, x) >= mag.at<float>(y - 1, x) &&
                        mag.at<float>(y, x) > mag.at<float>(y + 1, x));
                }
                if (ok) anchors.emplace_back(x, y);
            }
        }

        cv::Mat visited = cv::Mat::zeros(g8.size(), CV_8U);
        segments.clear();
        scores.clear();

        for (size_t i = 0; i < anchors.size(); ++i) {
            const cv::Point& a = anchors[i];
            if (visited.at<uint8_t>(a.y, a.x)) continue;
            std::vector<cv::Point> chain = draw_from_anchor(a.x, a.y, qori.at<uint8_t>(a.y, a.x),
                gx, gy, mag, visited, params_);
            if (!chain.empty()) {
                validate_and_split(chain, gx, gy, params_, segments, scores);
            }
        }
    }

    cv::Mat ELSED::drawSegments(const cv::Mat& img,
        const std::vector<Segment>& segs,
        const cv::Scalar& color) const
    {
        cv::Mat vis;
        if (img.channels() == 1) cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);
        else vis = img.clone();
        for (size_t i = 0; i < segs.size(); ++i) {
            const Segment& s = segs[i];
            cv::line(vis,
                cv::Point(cvRound(s.x1), cvRound(s.y1)),
                cv::Point(cvRound(s.x2), cvRound(s.y2)),
                color, 1, cv::LINE_AA);
        }
        return vis;
    }

} 
