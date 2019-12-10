#include <ctime>
#include <stack>
#include <tuple>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <opencv2/viz/vizcore.hpp>

namespace fs = boost::filesystem;


/* Support structures used by Ceres. */
struct Camera
{
    Camera() = default;
    Camera(const cv::Point3d &c) : c(c), b{} { b.fill(0.0); }
    cv::Point3d c;
    std::array<double, 4> b;
};
struct Point
{
    Point() : b{} { b.fill(0.0); }
    std::array<double, 3> b;
};

struct Observation
{
    Observation(std::shared_ptr<Camera> cam, cv::Point2d ip) : cam(cam), ip(ip) {}
    Observation& operator=(const Observation&) = default;
    std::shared_ptr<Camera> cam;
    cv::Point2d ip;
};

struct Track
{
    Track() : p(std::make_shared<Point>()), obs{} {}
    Track& operator=(const Track&) = default;
    std::shared_ptr<Point> p;
    std::vector<Observation> obs;
};


/** @brief Ceres residual class for bundle adjustment. */
struct Residual
{
    Residual(cv::Point3d c, cv::Point2d p_im) : c(c), p_im(p_im) {}

    template <typename T>
    bool operator()(const T * const cam,
                    const T * const point,
                    T *residual) const
    {
        // The camera is a 4 element pointer: [rx, ry, rz, f].
        // i.e., the rotation and focal distance.

        // The point is 3 element pointer: [x, y, z], the 3D location.
        T p[3];
        ceres::AngleAxisRotatePoint(cam, point, p);

        // Translate the point.
        p[0] += c.x;
        p[1] += c.y;
        p[2] += c.z;

        // Find the center of distortion and change the camera coordinate to
        // look down the negative z-axis, OpenGL style.
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Perform perspective correction.
        const T& f = cam[3];
        T predicted_x = f * xp;
        T predicted_y = f * yp;

        residual[0] = predicted_x - p_im.x;
        residual[1] = predicted_y - p_im.y;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(cv::Point3d c, cv::Point2d p)
    {
        constexpr size_t nresiduals = 2;
        constexpr size_t cam_sz = 4;
        constexpr size_t point_sz = 3;
        return (new ceres::AutoDiffCostFunction
                <Residual, nresiduals, cam_sz, point_sz>
                (new Residual(c, p)));
    }

    cv::Point3d c;
    cv::Point2d p_im;
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

std::tuple<std::vector<cv::Mat>, std::vector<cv::Point3d>, CameraGraph>
parse_camera_graph(const cv::String &path);


std::tuple<std::vector<std::shared_ptr<Camera>>, std::vector<Track>>
find_camera_tracks(
    const std::vector<cv::Mat> &images,
    const std::vector<cv::Point3d> &locations,
    const std::vector<std::vector<cv::KeyPoint>> &keypoints,
    const std::vector<std::vector<cv::DMatch>> &matches);

std::tuple<std::vector<double>,
           std::vector<cv::Vec3f>,
           std::vector<cv::Vec3f>,
           std::vector<cv::Vec3b>>
calibrate_cameras(const std::vector<cv::Mat>&,
                  const std::vector<cv::Point3d>&,
                  const CameraGraph &);

std::tuple<std::vector<double>,
           std::vector<cv::Vec3f>,
           std::vector<cv::Vec3f>,
           std::vector<cv::Vec3b>>
extract_sfm(const std::vector<std::shared_ptr<Camera>> &cameras,
            const std::vector<Track> &tracks);


bool
output_camera_calibration(const cv::String &path,
                          const std::vector<double> &focals,
                          const std::vector<cv::Vec3f> &poses);


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
    std::vector<cv::Point3d> locations;
    CameraGraph G;
    std::tie(images, locations, G) = parse_camera_graph(cam_graph);
    if (images.empty() || locations.empty() || G.empty())
    {
        std::cerr << "Unable to parse camera graph." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<double> focals;
    std::vector<cv::Vec3f> poses;
    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3b> colors;
    std::tie(focals, poses, points, colors) = calibrate_cameras(images, locations, G);
    if (focals.empty() || poses.empty() || points.empty() || colors.empty())
    {
        std::cerr << "Unable to calibrate cameras." << std::endl;
        return EXIT_FAILURE;
    }

    cv::viz::writeCloud(sfm_output, points, colors, cv::noArray(), false);
    if (!output_camera_calibration(cam_output, focals, poses))
    {
        std::cerr << "Unable to write camera calibration output." << std::endl;
        return EXIT_SUCCESS;
    }

    return EXIT_SUCCESS;
}


std::tuple<std::vector<cv::Mat>, std::vector<cv::Point3d>, CameraGraph>
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

    std::unordered_map<size_t, std::pair<cv::String, cv::Point3d>> vp;
    for (const auto &v : vertices)
    {
        size_t index = static_cast<int>(v["index"]);
        cv::String image = static_cast<cv::String>(v["image"]);
        cv::FileNode loc = v["location"];
        if (loc.type() != cv::FileNode::SEQ && loc.size() != 3)
        {
            std::cerr << "Missing or incorrect camera location." << std::endl;
            return {};
        }
        cv::Point3d cl = cv::Point3d(static_cast<double>(loc[0]),
                                     static_cast<double>(loc[1]),
                                     static_cast<double>(loc[2]));
        if (vp.count(index) == 1)
        {
            std::cerr << "Duplicate camera index." << std::endl;
            return {};
        }
        vp[index] = std::make_pair(image, cl);
    }

    // Rearrange to make indices match the vector index.
    size_t N = vp.size();
    std::vector<cv::Mat> images(N);
    std::vector<cv::Point3d> locations(N);
    for (size_t i = 0; i < N; ++i)
    {
        if (vp.count(i) != 1)
        {
            std::cerr << "Invalid camera enumeration." << std::endl;
            return {};
        }
        else
        {
            fs::path ip = dir / vp[i].first;
            images[i] = cv::imread(ip.string());
            locations[i] = vp[i].second;
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
    return std::make_tuple(images, locations, G);
}


std::tuple<std::vector<double>,
           std::vector<cv::Vec3f>,
           std::vector<cv::Vec3f>,
           std::vector<cv::Vec3b>>
calibrate_cameras(const std::vector<cv::Mat> &images,
                  const std::vector<cv::Point3d> &locations,
                  const CameraGraph &G)
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
    size_t N2 = N * N;
    // size_t W = N * (N - 1) / 2;
    std::vector<std::vector<cv::KeyPoint>> keypoints(N);
    std::vector<cv::Mat> desc(N);
    for (size_t i = 0; i < N; ++i)
    {
        detector->detect(images[i], keypoints[i]);
        detector->compute(images[i], keypoints[i], desc[i]);

        // cv::Mat out;
        // cv::drawKeypoints(images[i], keypoints[i], out);
        // cv::imshow("DebugWindow", out);
        // cv::waitKey(0);
    }

    // Search for matches in each pair of images.
    // Note: Crosscheck is intended as an alternative to the ratio test.
    std::vector<std::vector<cv::DMatch>> matches(N2);
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    for (size_t i = 0, cnt = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            if (i == j) continue;
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(desc[i], desc[j], knn_matches, 2);

            // Filter matches using Lowe's ratio test.
            const float nn_ration = 0.85;
            for (size_t k = 0; k < knn_matches.size(); k++)
            {
                if (knn_matches[k][0].distance < nn_ration * knn_matches[k][1].distance)
                {
                    matches[cnt].push_back(knn_matches[k][0]);
                }
            }

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

            // Estimate the fundamental matrix and use RanSaC filter out matches
            // further.
            std::vector<uchar> mask(matches[cnt].size());
            cv::Mat F = cv::findFundamentalMat(pt0, pt1, cv::FM_RANSAC, 3.0, 0.9999, mask);

            // Filter out the outlier matches.
            std::vector<cv::DMatch> rmatch;
            for (size_t k = 0; k < matches[cnt].size(); ++k)
            {
                if (k < mask.size() && mask[k])
                {
                    rmatch.push_back(matches[cnt][k]);
                }
            }

            // Display the filtered matches.
            // cv::Mat out;
            // cv::drawMatches(images[i], keypoints[i], images[j], keypoints[j],
            //                 rmatch, out,
            //                 cv::Scalar::all(-1), cv::Scalar::all(-1),
            //                 std::vector<char>(),
            //                 cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            // cv::imshow("DebugWindow", out);
            // cv::waitKey(0);

            // Replace matches[cnt] with our filtered matches.
            matches[cnt] = rmatch;
            cnt++;
        }
    }

    // Initialize the Bundle Adjustment problem and set up a first estimate for
    // the camera parameters and SFM by finding tracks among the matches - i.e.,
    // cycles of matches among the images.
    std::vector<std::shared_ptr<Camera>> cameras;
    std::vector<Track> tracks;
    std::tie(cameras, tracks) = find_camera_tracks(images, locations, keypoints, matches);

    // Create residuals and add them to Ceres.
    ceres::Problem problem;
    for (auto &t : tracks)
    {
        for (auto &o : t.obs)
        {
            ceres::CostFunction *cf = Residual::Create(o.cam->c, o.ip);
            problem.AddResidualBlock(cf, nullptr,
                                     o.cam->b.data(),
                                     t.p->b.data());
        }
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    return extract_sfm(cameras, tracks);
}



struct Edge
{
    bool operator==(const Edge &rhs) const
    {
        return std::make_tuple(src_img, dst_img, kp) ==
            std::make_tuple(rhs.src_img, rhs.dst_img, rhs.kp);
    }

    struct Hash
    {
        size_t operator()(const Edge &e) const
        {
            return e.src_img ^ e.dst_img ^ e.kp;
        }
    };

    size_t src_img; /**< Edge source image. */
    size_t dst_img; /**< Edge destination image. */
    size_t kp;      /**< Keypoint index. */
};

struct Vertex
{
    size_t img; /**< Source image. */
    size_t kp;  /**< Keypoint index. */
};

// using ImageGraph = std::unordered_map<Edge, Vertex, Edge::Hash>;
using ImageGraph = std::unordered_map<size_t, std::vector<size_t>>;


ImageGraph filter_graph(const ImageGraph &G);

std::ostream &write_dotfile(std::ostream &os,
                            const ImageGraph &G,
                            const std::vector<cv::KeyPoint> &kps,
                            const std::vector<std::unordered_set<size_t>> &immap)
{
    // Do a quick count of the number of immediate edges lead to each vertex.
    std::vector<size_t> cnt(kps.size());
    for (auto &p : G)
    {
        for (auto &e : p.second)
        {
            cnt[e]++;
        }
    }

    // Dump the image graph as a dot-file.
    os << "digraph G {" << "\n";
    size_t N = immap.size();
    for (size_t i = 0; i < N; ++i)
    {
        os << "subgraph cluster_" << i << " {" << "\n";
        std::vector<size_t> v(immap[i].begin(), immap[i].end());
        std::sort(v.begin(), v.end());
        for (const auto &ki : v)
        {
            if (cnt[ki] > 0 || !G.at(ki).empty())
            {
                const auto &k = kps[ki];
                os << ki
                   << "[label=\"" << k.pt << "\"" << "]"
                   << ";" << "\n";
            }
        }
        os << "}" << "\n";
    }

    for (auto &p : G)
    {
        for (auto &e : p.second)
        {
            os << p.first << "->" << e << ";" << "\n";
        }
    }
    os << "}" << std::endl;

    return os;
}


cv::Mat visualize_image_graph(const ImageGraph &G,
                              const std::vector<cv::KeyPoint> &kps,
                              const std::vector<std::unordered_set<size_t>> &immap,
                              const std::vector<cv::Mat> &images)
{
    size_t x = images[0].cols;
    size_t y = images[0].rows;
    size_t N = images.size();
    bool is_odd = (N % 2 != 0);

    std::vector<cv::Point2i> off(N);
    size_t ri = N / 2 + is_odd;

    std::vector<cv::Mat> imgs(N + is_odd);
    for (size_t i = 0; i < N; ++i)
    {
        cv::resize(images[i], imgs[i], cv::Size(x, y));
    }
    if (is_odd)
    {
        imgs[N] = cv::Mat(x, y, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    // Compute image origin offsets.
    for (size_t i = 0; i < ri; ++i)
    {
        off[i] = cv::Point2i(i * x, 0);
    }
    for (size_t i = ri; i < N; ++i)
    {
        off[i] = cv::Point2i((i - ri) * x, y);
    }

    // Concatenate all images.
    cv::Mat row0;
    cv::Mat row1;
    cv::hconcat(&imgs[0], ri, row0);
    cv::hconcat(&imgs[ri], imgs.size() - ri, row1);
    cv::Mat out;
    cv::vconcat(row0, row1, out);

    // Create a colormap that we can look up in.
    cv::Mat cmap_in(1, N, CV_8UC1, cv::Scalar(0));
    for (size_t i = 0; i < N; ++i)
    {
        const int step = 255 / N;
        cmap_in.at<uchar>(0, i) = step * i;
    }
    cv::Mat cmap;
    cv::applyColorMap(cmap_in, cmap, cv::COLORMAP_VIRIDIS);


    // Map all keypoints to our concatenated image.
    std::vector<cv::Point2d> mp(kps.size());
    std::vector<cv::Vec3b> kpcolor(kps.size());
    for (size_t i = 0; i < N; ++i)
    {
        for (const auto &ki : immap[i])
        {
            const auto &k = kps[ki];
            mp[ki] = cv::Point2d(k.pt.x, k.pt.y) + cv::Point2d(off[i]);
            kpcolor[ki] = cmap.at<cv::Vec3b>(0, i);
        }
    }

    // Draw the keypoints and arrows between them according to our graph.
    for (auto &p : G)
    {
        const auto &ki = p.first;
        const auto &src = mp[ki];
        if (p.second.size() > 0)
        {
            cv::circle(out, src, 3, kpcolor[ki]);
            for (auto &e : p.second)
            {
                const auto &dst = mp[e];
                cv::arrowedLine(out, src, dst, kpcolor[ki], 1, 8, 0, 0.005);
            }
        }
        else
        {
            cv::circle(out, src, 3, CV_RGB(255, 0, 0));
        }
    }
    return out;
}


std::tuple<std::vector<std::shared_ptr<Camera>>, std::vector<Track>>
find_camera_tracks(
    const std::vector<cv::Mat> &images,
    const std::vector<cv::Point3d> &locations,
    const std::vector<std::vector<cv::KeyPoint>> &keypoints,
    const std::vector<std::vector<cv::DMatch>> &matches)
{
    // Build image graph over all matches.
    size_t N = keypoints.size();
    std::vector<std::unordered_set<size_t>> immap(N);
    std::vector<cv::KeyPoint> fkps;   // Flattened keypoints.
    std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> fmap;

    ImageGraph G;
    for (size_t i = 0, cnt = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            if (i == j) continue;

            // Create edges for all matches.
            for (const auto &m : matches[cnt])
            {
                // Find or create indices for the current pair of keypoints.
                auto si = 0;
                auto di = 0;

                size_t qi = m.queryIdx;
                size_t ti = m.trainIdx;

                auto src = fmap.find(std::make_pair(i, qi));
                auto dst = fmap.find(std::make_pair(j, ti));
                if (src == fmap.end())
                {
                    cv::KeyPoint kp = keypoints[i][qi];
                    fkps.push_back(kp);
                    si = fkps.size() - 1;
                    immap[i].insert(si);
                    fmap.emplace(std::make_pair(i, qi), si);
                }
                else
                {
                    si = src->second;
                }
                if (dst == fmap.end())
                {
                    cv::KeyPoint kp = keypoints[j][ti];
                    fkps.push_back(kp);
                    di = fkps.size() - 1;
                    immap[j].insert(di);
                    fmap.emplace(std::make_pair(j, ti), di);
                }
                else
                {
                    di = dst->second;
                }

                G[si].push_back(di);
                if (G.count(di) == 0)
                {
                    G[di];
                }
            }
            cnt++;
        }
    }

    // Perform graph filtering.
    cv::Mat ig = visualize_image_graph(G, fkps, immap, images);
    std::ofstream f0("image_graph.dot");
    write_dotfile(f0, G, fkps, immap);
    cv::imwrite("image_graph.jpg", ig);

    G = filter_graph(G);

    cv::Mat fig = visualize_image_graph(G, fkps, immap, images);
    std::ofstream f1("filtered_image_graph.dot");
    write_dotfile(f1, G, fkps, immap);
    cv::imwrite("filtered_image_graph.jpg", fig);

    // TODO: More and/or better filtering is possible.

    // cv::imshow("DebugWindow", fig);
    // cv::waitKey(0);

    // Extract 'tracks' from the camera graphs. I.e., any cycles found in the
    // graph.
    std::vector<std::shared_ptr<Camera>> cameras;
    std::vector<Track> tracks;
    for (auto &l : locations)
    {
        cameras.push_back(std::make_shared<Camera>(l));
    }

    for (auto &p : G)
    {
        const auto &ki = p.first;

        // Find the associated camera.
        size_t imgi = N + 1;
        for (size_t i = 0; i < N; ++i)
        {
            if (immap[i].find(ki) != immap[i].end())
            {
                imgi = i;
                break;
            }
        }
        if (imgi > N)
        {
            continue;
        }

        // Create a track for this key-point.
        if (!p.second.empty())
        {
            const auto &kp = fkps[ki];
            const auto &cam = cameras[imgi];
            Track tr;
            tr.obs.push_back(Observation(cam, cv::Point2d(kp.pt)));
            for (auto &e : p.second)
            {
                tr.obs.push_back(Observation(cam, cv::Point2d(fkps[e].pt)));
            }
            tracks.push_back(tr);
        }
    }

    return std::make_tuple(cameras, tracks);
}


ImageGraph filter_graph(const ImageGraph &G)
{
    std::vector<size_t> cnt(G.size());

    // Run DFS, count the number of visits a vertex receives.
    for (auto &p : G)
    {
        std::stack<size_t> s{{p.first}};
        std::vector<bool> visited(G.size(), false);
        while (!s.empty())
        {
            auto &v = s.top(); s.pop();
            visited[v] = true;
            cnt[v]++;
            for (auto &e : G.at(v))
            {
                if (!visited[e])
                {
                    s.push(e);
                }
            }
        }
    }

    // Remove all vertices that weren't visited.
    ImageGraph N(G);
    for (auto &p : G)
    {
        auto &ki = p.first;
        auto &edges = p.second;
        if (cnt[ki] == 1 && edges.size() == 1)
        {
            auto &e = edges[0];
            if (cnt[e] == 2)
            {
                N[ki].clear();
            }
        }
    }

    return N;
}


std::tuple<std::vector<double>,
           std::vector<cv::Vec3f>,
           std::vector<cv::Vec3f>,
           std::vector<cv::Vec3b>>
extract_sfm(const std::vector<std::shared_ptr<Camera>> &cameras,
            const std::vector<Track> &tracks)
{
    std::vector<double> focals;
    std::vector<cv::Vec3f> rot;
    std::vector<cv::Vec3f> point_cloud;
    std::vector<cv::Vec3b> point_colors;
    for (auto &cam : cameras)
    {
        focals.push_back(cam->b[3]);
        rot.push_back(cv::Vec3f(cam->b[0], cam->b[1], cam->b[2]));
        point_cloud.push_back(cv::Vec3f(cam->c.x, cam->c.y, cam->c.z));
        point_colors.push_back(cv::Vec3f(0, 255, 0));
    }

    for (auto &tr : tracks)
    {
        point_cloud.push_back(cv::Vec3f(tr.p->b[0], tr.p->b[1], tr.p->b[2]));
        point_colors.push_back(cv::Vec3f(0, 0, 255));
    }

    return std::make_tuple(focals, rot, point_cloud, point_colors);
}



bool
output_camera_calibration(const cv::String &path,
                          const std::vector<double> &focals,
                          const std::vector<cv::Vec3f> &poses)
{
    std::time_t date = time(nullptr);
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "ncamera" << static_cast<int>(focals.size());
    fs << "calibration_date" << std::asctime(std::localtime(&date));
    fs << "camera_focals" << "[";
    for (const auto &f : focals)
    {
        fs << f << ",";
    }
    fs << "]";
    fs << "camera_poses" << "[";
    for (const auto &p : poses)
    {
        fs << p;
    }
    fs << "]";
    fs.release();
    return true;
}
