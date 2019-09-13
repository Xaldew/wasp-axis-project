#include <ctime>
#include <tuple>
#include <iostream>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <opencv2/viz/vizcore.hpp>

namespace fs = boost::filesystem;

/** @brief Ceres residual class for bundle adjustment. */
struct Residual
{
    Residual(cv::Point2d im0, cv::Point2d im1) : im0(im0), im1(im1) {}

    template <typename T>
    bool operator()(const T * const cam0,
                    const T * const cam1,
                    const T * const point,
                    T *residual) const
    {
        // The camera is a 7 element pointer: [rx, ry, rz, tx, ty, tz, f].
        // The point is 3D pointer: [x, y, z].
        T p0[3];
        ceres::AngleAxisRotatePoint(cam0, point, p0);

        p0[0] += cam0[3];
        p0[1] += cam0[4];
        p0[2] += cam0[5];

        T xp0 = -p0[0] / p0[2];
        T yp0 = -p0[1] / p0[2];

        residual[0] = cam0[6] * xp0 - im0.x;
        residual[1] = cam0[6] * yp0 - im0.y;

        T p1[3];
        ceres::AngleAxisRotatePoint(cam1, point, p1);

        p1[0] += cam1[3];
        p1[1] += cam1[4];
        p1[2] += cam1[5];

        T xp1 = -p1[0] / p1[2];
        T yp1 = -p1[1] / p1[2];

        residual[2] = cam1[6] * xp1 - im1.x;
        residual[3] = cam1[6] * yp1 - im1.y;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(cv::Point2d im0, cv::Point2d im1)
    {
        constexpr size_t nresiduals = 4;
        constexpr size_t cam_sz = 7;
        constexpr size_t point_sz = 3;
        return (new ceres::AutoDiffCostFunction
                <Residual, nresiduals, cam_sz, cam_sz, point_sz>
                (new Residual(im0, im1)));
    }

    cv::Point2d im0;
    cv::Point2d im1;
};

struct pair_hash
{
    template<class T0, class T1>
    std::size_t operator()(const std::pair<T0, T1> &p) const
    {
        return std::hash<T0>()(p.first) ^ std::hash<T1>()(p.second);
    }
};

using CameraGraph = std::unordered_map<std::pair<std::size_t, size_t>, float, pair_hash>;

std::pair<std::vector<cv::Mat>, CameraGraph>
parse_camera_graph(const cv::String &path);

std::tuple<std::vector<cv::Matx33f>,
           std::vector<cv::Matx13f>,
           std::vector<cv::Matx13f>>
calibrate_cameras(const CameraGraph &G, const std::vector<cv::Mat> &images);

bool output_camera_calibration(const cv::String &path,
                               const std::vector<cv::Matx33f> &camera_matrices);


int main(int argc, char *argv[])
{
    const cv::String help_text =
        "{ help h usage ? |        | Print this message. }"
        "{ @camera_graph  | <none> | Path to camera configuration graph. }"
        "{ @camera_output | <none> | Path for camera calibgration output. }"
        "{ @sfm_output    | <none> | Path for 3D point cloud output. }";
    cv::CommandLineParser parser(argc, argv, help_text);
    parser.about("This is multi camera calibration and "
                 "structure from motion estimation program.");

    cv::String cam_graph = parser.get<cv::String>("@camera_graph");
    cv::String cam_output = parser.get<cv::String>("@camera_output");
    cv::String sfm_output = parser.get<cv::String>("@sfm_output");

    if (!parser.check())
    {
        parser.printErrors();
        return EXIT_FAILURE;
    }
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    std::vector<cv::Mat> images;
    CameraGraph G;
    std::tie(images, G) = parse_camera_graph(cam_graph);
    if (images.empty() || G.empty())
    {
        std::cerr << "Unable to parse camera graph." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<cv::Matx33f> camera_matrices;
    std::vector<cv::Matx13f> points;
    std::vector<cv::Matx13f> colors;
    std::tie(camera_matrices, points, colors) = calibrate_cameras(G, images);
    if (camera_matrices.empty() || points.empty() || colors.empty())
    {
        std::cerr << "Unable to calibrate cameras." << std::endl;
        return EXIT_FAILURE;
    }

    cv::viz::writeCloud(sfm_output, points, colors, cv::noArray(), false);
    if (!output_camera_calibration(cam_output, camera_matrices))
    {
        std::cerr << "Unable to write camera calibration output." << std::endl;
        return EXIT_SUCCESS;
    }

    return EXIT_SUCCESS;
}


std::pair<std::vector<cv::Mat>, CameraGraph>
parse_camera_graph(const cv::String &path)
{
    fs::path dir = fs::path(path).parent_path();
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        return {};
    }

    cv::FileNode vertices = fs["vertices"];
    cv::FileNode edges = fs["edges"];
    if (vertices.type() != cv::FileNode::SEQ || edges.type() != cv::FileNode::SEQ)
    {
        return {};
    }

    std::unordered_map<size_t, cv::String> vp;
    for (const auto &v : vertices)
    {
        size_t index = static_cast<int>(v["index"]);
        cv::String image = static_cast<cv::String>(v["image"]);
        if (vp.count(index) == 1)
        {
            std::cerr << "Duplicate camera index." << std::endl;
            return {};
        }
        vp[index] = image;
    }

    // Rearrange to make indices match the vector index.
    size_t N = vp.size();
    std::vector<cv::Mat> images(N);
    for (size_t i = 0; i < N; ++i)
    {
        if (vp.count(i) != 1)
        {
            std::cerr << "Invalid camera enumeration." << std::endl;
            return {};
        }
        else
        {
            fs::path ip = dir / vp[i];
            images[i] = cv::imread(ip.string());
        }
    }

    CameraGraph G;
    for (const auto &e : edges)
    {
        if (e.type() != cv::FileNode::SEQ && e.size() != 3)
        {
            std::cerr << "Invalid edge specification." << std::endl;
            return {};
        }
        size_t u = static_cast<int>(e[0]);
        size_t v = static_cast<int>(e[1]);
        float w = static_cast<float>(e[2]);
        if (vp.count(u) != 1 || vp.count(v) != 1)
        {
            std::cerr << "Invalid edge: "
                      << u << " -> " << v << " (" << w << ")" << std::endl;
            return {};
        }
        G[std::make_pair(u, v)] = w;
    }
    return std::make_pair(images, G);
}


std::tuple<std::vector<cv::Matx33f>,
           std::vector<cv::Matx13f>,
           std::vector<cv::Matx13f>>
calibrate_cameras(const CameraGraph &G, const std::vector<cv::Mat> &images)
{
    // DEBUG: Print graph.
    for (auto &p : G)
    {
        std::cout << p.first.first << " " << p.first.second << " " << p.second << "\n";
    }
    // DEBUG: Display images.
    cv::namedWindow("DebugWindow", cv::WINDOW_AUTOSIZE);

    // Detect features in all images.
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    size_t N = images.size();
    size_t W = N * (N - 1) / 2;
    std::vector<std::vector<cv::KeyPoint>> keypoints(N);
    std::vector<cv::Mat> desc(N);
    for (size_t i = 0; i < N; ++i)
    {
        detector->detect(images[i], keypoints[i]);
        detector->compute(images[i], keypoints[i], desc[i]);
    }

    // Search for matches in each pair of images.
    std::vector<std::vector<cv::DMatch>> matches(W);
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    for (size_t i = 0, cnt = 0; i < N; ++i)
    {
        for (size_t j = i + 1; j < N; ++j, ++cnt)
        {
            matcher.match(desc[i], desc[j], matches[cnt]);

            // Extract indices to the keypoints.
            std::vector<int> id0;
            std::vector<int> id1;
            for (const auto &m : matches[cnt])
            {
                id0.push_back(m.queryIdx);
                id1.push_back(m.trainIdx);
            }

            // Extract the correspondence image points.
            std::vector<cv::Point2f> pt0;
            cv::KeyPoint::convert(keypoints[i], pt0, id0);
            std::vector<cv::Point2f> pt1;
            cv::KeyPoint::convert(keypoints[j], pt1, id1);

            cv::Mat mask;
            cv::Mat F = cv::findFundamentalMat(pt0, pt1, cv::FM_RANSAC, 3.0, 0.9999, mask);

            // Filter out the outlier matches.
            std::vector<cv::DMatch> rmatch;
            for (size_t k = 0; k < matches[cnt].size(); ++k)
            {
                if (mask.at<int>(k))
                {
                    rmatch.push_back(matches[cnt][k]);
                }
            }

            // Display the filtered matches.
            cv::Mat out;
            cv::drawMatches(images[i], keypoints[i], images[j], keypoints[j],
                            matches[cnt], out,
                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                            std::vector<char>(),
                            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::imshow("DebugWindow", out);
            cv::waitKey(0);
        }
    }

    size_t cloud_sz = 0;
    for (auto &m : matches) { cloud_sz += m.size(); }

    // Note: The matches are arguably too poor to work well in this example.

    // Use Ceres to estimate the bundle adjustment for all correspondences.
    struct Camera
    {
        double p[7];
    };
    struct Point
    {
        double p[3];
    };
    using namespace std;

    std::vector<Camera> cameras(N);
    std::vector<Point> sfm(cloud_sz);
    ceres::Problem problem;
    size_t pidx = 0;
    for (size_t i = 0, cnt = 0; i < N; ++i)
    {
        for (size_t j = i + 1; j < N; ++j, ++cnt)
        {
            for (auto &m : matches[cnt])
            {
                cv::Point2d im0 = keypoints[i][m.queryIdx].pt;
                cv::Point2d im1 = keypoints[j][m.trainIdx].pt;
                cout << im0 << " " << im1 << endl;
                ceres::CostFunction *cf = Residual::Create(im0, im1);
                problem.AddResidualBlock(cf, nullptr,
                                         cameras[i].p,
                                         cameras[j].p,
                                         sfm[pidx++].p);
            }
        }
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    return {};
}


bool output_camera_calibration(const cv::String &path,
                               const std::vector<cv::Matx33f> &camera_matrices)
{
    std::time_t date = time(nullptr);
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "ncamera" << static_cast<int>(camera_matrices.size());
    fs << "calibration_date" << std::asctime(std::localtime(&date));
    fs << "camera_calibration" << "[";
    for (const auto &c : camera_matrices)
    {
        fs << c;
    }
    fs << "]";
    fs.release();
    return true;
}
