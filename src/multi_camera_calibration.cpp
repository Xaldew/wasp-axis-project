#include <ctime>
#include <tuple>
#include <iostream>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/viz/vizcore.hpp>


namespace fs = boost::filesystem;


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
    for (auto &im : images)
    {
        cv::imshow("DebugWindow", im);
        cv::waitKey(0);
    }

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
