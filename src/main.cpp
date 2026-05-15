/**
 * @file main.cpp
 * @brief 机器人装甲板检测主程序
 *
 * 功能：
 *  - 使用海康 MVS SDK 连接相机采集图像
 *  - 使用 OpenVINO 加载 ONNX 模型进行推理（参考 OpenvinoInfer.cpp 中的预处理/后处理方法）
 *  - 绘制检测结果并实时显示
 *
 * 参考了 auto_aim 项目中的 Camera 类（海康相机驱动）和 OpenvinoInfer 推理管线
 */

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "OpenvinoInfer.h"

#include <MvCameraControl.h>

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <cstring>
#include <csignal>

using namespace cv;
using namespace std;

// ============================================================
// 全局变量
// ============================================================
static bool g_bExit = false;
void signalHandler(int) { g_bExit = true; }

// ============================================================
// 工具函数：将 .onnx 模型转换为 OpenVINO IR 格式 (.xml + .bin)
// 这样可以使用 OpenvinoInfer 的第一个构造函数（xml_path, bin_path, device）
// ============================================================
std::pair<string, string> convertOnnxToIR(const string& onnx_path) {
    ov::Core core;
    cout << "[INFO] Loading ONNX model: " << onnx_path << endl;
    auto model = core.read_model(onnx_path);

    string base_path = onnx_path;
    // 去掉 .onnx 后缀
    size_t dot_pos = base_path.rfind(".onnx");
    if (dot_pos != string::npos) {
        base_path = base_path.substr(0, dot_pos);
    }

    string xml_path = base_path + ".xml";
    string bin_path = base_path + ".bin";

    cout << "[INFO] Serializing model to IR format..." << endl;
    ov::serialize(model, xml_path, bin_path);
    cout << "[INFO] IR files generated: " << xml_path << ", " << bin_path << endl;

    return {xml_path, bin_path};
}

// ============================================================
// 类别名称映射（对应 README 中的标签定义）
// ============================================================
static const char* CLASS_NAMES[] = {
    "Sentry",  // 0: G   - 哨兵
    "1",       // 1: 1   - 一号
    "2",       // 2: 2   - 二号
    "3",       // 3: 3   - 三号
    "4",       // 4: 4   - 四号
    "5",       // 5: 5   - 五号
    "Outpost", // 6: O   - 前哨站
    "Base",    // 7: Bs  - 基地
    "BigBase"  // 8: Bb  - 基地大装甲
};

static const cv::Scalar COLOR_RED   = cv::Scalar(0, 0, 255);
static const cv::Scalar COLOR_BLUE  = cv::Scalar(255, 0, 0);
static const cv::Scalar COLOR_GREEN = cv::Scalar(0, 255, 0);
static const cv::Scalar COLOR_CYAN  = cv::Scalar(255, 255, 0);
static const cv::Scalar COLOR_WHITE = cv::Scalar(255, 255, 255);
static const cv::Scalar COLOR_BLACK = cv::Scalar(0, 0, 0);
static const cv::Scalar COLOR_GRAY  = cv::Scalar(200, 200, 200);

// ============================================================
// 绘制检测结果
// 包含：检测框(颜色按队伍)、置信度、分类标签、4个关键点编号、装甲板尺寸
// 参考 OpenvinoInfer::infer() 中的 Object 结构
// ============================================================
void drawDetectionResults(cv::Mat& image, const vector<Object>& objects) {
    // -------- 图例：显示颜色对应关系 --------
    cv::rectangle(image, cv::Rect(10, 120, 12, 12), COLOR_RED, -1);
    cv::putText(image, "RED Team", cv::Point(28, 132),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1);
    cv::rectangle(image, cv::Rect(10, 140, 12, 12), COLOR_BLUE, -1);
    cv::putText(image, "BLUE Team", cv::Point(28, 152),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 1);

    for (const auto& obj : objects) {
        // -------- 跳过无效检测框（面积为0或完全越界）--------
        if (obj.rect.width <= 0 || obj.rect.height <= 0 ||
            obj.rect.x >= image.cols || obj.rect.y >= image.rows ||
            obj.rect.x + obj.rect.width <= 0 || obj.rect.y + obj.rect.height <= 0) {
            continue;
        }

        // ================================================================
        // 1. 根据队伍颜色选择主题色
        // ================================================================
        // obj.color: 0 = RED, 1 = BLUE
        cv::Scalar team_color = (obj.color == 0) ? COLOR_RED :
                                (obj.color == 1) ? COLOR_BLUE : COLOR_GREEN;
        cv::Scalar team_color_dim = (obj.color == 0) ? cv::Scalar(0, 0, 180) :
                                    (obj.color == 1) ? cv::Scalar(180, 0, 0) : cv::Scalar(0, 180, 0);

        // ================================================================
        // 2. 绘制检测框（外边框 + 内角标）
        // ================================================================
        cv::rectangle(image, obj.rect, team_color, 2);

        // 在四个角绘制短角标增强视觉效果
        int corner_len = std::min(obj.rect.width, obj.rect.height) / 4;
        corner_len = std::max(corner_len, 10);
        // 左上角
        cv::line(image, cv::Point(obj.rect.x, obj.rect.y + corner_len),
                       cv::Point(obj.rect.x, obj.rect.y), team_color, 2);
        cv::line(image, cv::Point(obj.rect.x, obj.rect.y),
                       cv::Point(obj.rect.x + corner_len, obj.rect.y), team_color, 2);
        // 右上角
        cv::line(image, cv::Point(obj.rect.br().x - corner_len, obj.rect.y),
                       cv::Point(obj.rect.br().x, obj.rect.y), team_color, 2);
        cv::line(image, cv::Point(obj.rect.br().x, obj.rect.y),
                       cv::Point(obj.rect.br().x, obj.rect.y + corner_len), team_color, 2);
        // 左下角
        cv::line(image, cv::Point(obj.rect.x, obj.rect.br().y - corner_len),
                       cv::Point(obj.rect.x, obj.rect.br().y), team_color, 2);
        cv::line(image, cv::Point(obj.rect.x, obj.rect.br().y),
                       cv::Point(obj.rect.x + corner_len, obj.rect.br().y), team_color, 2);
        // 右下角
        cv::line(image, cv::Point(obj.rect.br().x - corner_len, obj.rect.br().y),
                       cv::Point(obj.rect.br().x, obj.rect.br().y), team_color, 2);
        cv::line(image, cv::Point(obj.rect.br().x, obj.rect.br().y),
                       cv::Point(obj.rect.br().x, obj.rect.br().y - corner_len), team_color, 2);

        // ================================================================
        // 3. 绘制四个关键点
        //    landmarks 原始顺序（逆时针）：TL, BL, BR, TR
        //    绘制顺序：TL -> TR -> BR -> BL（顺时针连线）
        // ================================================================
        cv::Point2f pts_clockwise[4];
        pts_clockwise[0] = cv::Point2f(obj.landmarks[0], obj.landmarks[1]);  // TL (左上)
        pts_clockwise[1] = cv::Point2f(obj.landmarks[6], obj.landmarks[7]);  // TR (右上)
        pts_clockwise[2] = cv::Point2f(obj.landmarks[4], obj.landmarks[5]);  // BR (右下)
        pts_clockwise[3] = cv::Point2f(obj.landmarks[2], obj.landmarks[3]);  // BL (左下)

        // 关键点颜色（按序号区分）
        static const cv::Scalar kpt_colors[4] = {
            cv::Scalar(0, 255, 255),   // #1 TL - 黄色
            cv::Scalar(255, 170, 0),   // #2 TR - 蓝色
            cv::Scalar(170, 255, 0),   // #3 BR - 粉紫色
            cv::Scalar(0, 170, 255)    // #4 BL - 橙色
        };

        // 用虚线/实线组合连接关键点形成装甲板轮廓
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            cv::line(image, pts_clockwise[i], pts_clockwise[j], team_color, 1, cv::LINE_AA);
        }

        // 绘制关键点圆点（实心 + 序号标注）
        for (int i = 0; i < 4; i++) {
            // 外发光效果（大半径半透明）
            cv::circle(image, pts_clockwise[i], 7, kpt_colors[i], -1, cv::LINE_AA);
            // 白色内圈
            cv::circle(image, pts_clockwise[i], 4, COLOR_WHITE, -1, cv::LINE_AA);
            // 黑色边框
            cv::circle(image, pts_clockwise[i], 4, COLOR_BLACK, 1, cv::LINE_AA);
            // 标注序号 (1~4)
            cv::putText(image, std::to_string(i + 1),
                        pts_clockwise[i] + cv::Point2f(6, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 2);
            cv::putText(image, std::to_string(i + 1),
                        pts_clockwise[i] + cv::Point2f(6, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, kpt_colors[i], 1);
        }

        // ================================================================
        // 4. 分类标签 + 置信度（带背景框）
        // ================================================================
        // 获取类别名称
        const char* class_name = (obj.label >= 0 && obj.label <= 8) ?
                                 CLASS_NAMES[obj.label] : "Unknown";

        // 标签文字：分类名 置信度(百分比)
        string label_str = cv::format("%s %.1f%%", class_name, obj.prob * 100.0);

        int baseline = 0;
        cv::Size label_sz = cv::getTextSize(label_str, cv::FONT_HERSHEY_SIMPLEX,
                                             0.6, 2, &baseline);
        // 背景框：在检测框上方
        cv::Rect label_bg(obj.rect.x,
                          obj.rect.y - label_sz.height - 8,
                          label_sz.width + 12,
                          label_sz.height + 8);

        // 如果背景框超出图像上边界，放到框内下侧
        if (label_bg.y < 0) {
            label_bg.y = obj.rect.y + obj.rect.height + 5;
        }

        // 将 label_bg 钳位到图像范围内，防止 ROI 越界
        int img_w = image.cols, img_h = image.rows;
        cv::Rect label_bg_clipped = label_bg;
        label_bg_clipped.x      = std::max(0, label_bg_clipped.x);
        label_bg_clipped.y      = std::max(0, label_bg_clipped.y);
        label_bg_clipped.width  = std::min(label_bg_clipped.width,  img_w - label_bg_clipped.x);
        label_bg_clipped.height = std::min(label_bg_clipped.height, img_h - label_bg_clipped.y);

        // 绘制半透明背景（仅在裁剪后尺寸有效时）
        if (label_bg_clipped.width > 0 && label_bg_clipped.height > 0) {
            cv::Mat roi = image(label_bg_clipped);
            cv::Mat overlay = roi.clone();
            cv::rectangle(overlay, cv::Rect(0, 0, label_bg_clipped.width, label_bg_clipped.height),
                          team_color_dim, -1);
            cv::addWeighted(overlay, 0.7, roi, 0.3, 0, roi);
        }

        // 背景边框
        cv::rectangle(image, label_bg, team_color, 1);

        // 文字
        cv::putText(image, label_str,
                    cv::Point(label_bg.x + 6, label_bg.y + label_sz.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2);
        cv::putText(image, label_str,
                    cv::Point(label_bg.x + 6, label_bg.y + label_sz.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, team_color, 1);

        // ================================================================
        // 5. 队伍颜色标记 + 编号
        // ================================================================
        string color_tag = (obj.color == 0) ? "RED" : (obj.color == 1) ? "BLUE" : "?";
        string id_str = cv::format("[%s] #%d", color_tag.c_str(), obj.label);

        cv::putText(image, id_str,
                    cv::Point(obj.rect.x + 5, obj.rect.y + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1);
        cv::putText(image, id_str,
                    cv::Point(obj.rect.x + 5, obj.rect.y + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, team_color, 1);

        // ================================================================
        // 6. 装甲板尺寸信息（在框下方）
        // ================================================================
        int info_y = obj.rect.y + obj.rect.height + 15;
        // 如果标签已占用下方位置，下移
        if (label_bg.y > obj.rect.y) {
            info_y = label_bg.y + label_bg.height + 5;
        }

        // 尺寸行：WxH
        string size_str = cv::format("%.0f x %.0f", obj.rect.width, obj.rect.height);
        cv::putText(image, size_str,
                    cv::Point(obj.rect.x, info_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1);

        // 宽高比行：R = length/width, L, W
        string ratio_str = cv::format("R:%.2f L:%.0f W:%.0f",
                                      obj.ratio, obj.length, obj.width);
        cv::putText(image, ratio_str,
                    cv::Point(obj.rect.x, info_y + 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1);

        // ================================================================
        // 7. 置信度指示条（检测框下方的迷你进度条）
        // ================================================================
        int bar_width = std::min((int)obj.rect.width, 80);
        int bar_height = 4;
        int bar_x = obj.rect.x;
        int bar_y = obj.rect.y + obj.rect.height + 35;
        if (bar_y < 0) bar_y = obj.rect.y - 15;

        // 钳位进度条位置到图像范围内
        bar_y = std::max(0, std::min(bar_y, img_h - bar_height));
        bar_x = std::max(0, std::min(bar_x, img_w - bar_width));

        // 背景条（灰色）
        if (bar_y + bar_height <= img_h && bar_x + bar_width <= img_w) {
            cv::rectangle(image, cv::Rect(bar_x, bar_y, bar_width, bar_height),
                          cv::Scalar(100, 100, 100), -1);
        }
        // 前景条（按置信度填充，颜色随队伍）
        int fill_w = (int)(bar_width * obj.prob);
        if (fill_w > 0 && bar_y + bar_height <= img_h && bar_x + bar_width <= img_w) {
            cv::rectangle(image, cv::Rect(bar_x, bar_y, fill_w, bar_height),
                          team_color, -1);
        }
        // 置信度百分比文字（在条上方或右侧）
        string conf_pct = cv::format("%.0f%%", obj.prob * 100.0);
        cv::putText(image, conf_pct,
                    cv::Point(bar_x + bar_width + 4, bar_y + bar_height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, team_color, 1);

        // ================================================================
        // 8. 装甲板中心点十字标记
        // ================================================================
        cv::Point2f center(obj.rect.x + obj.rect.width / 2.0,
                           obj.rect.y + obj.rect.height / 2.0);
        int cross_len = 6;
        cv::line(image, cv::Point(center.x - cross_len, center.y),
                       cv::Point(center.x + cross_len, center.y),
                       team_color, 1);
        cv::line(image, cv::Point(center.x, center.y - cross_len),
                       cv::Point(center.x, center.y + cross_len),
                       team_color, 1);
    }
}

// ============================================================
// 海康相机图像处理：将原始相机数据转换为 OpenCV Mat
// 参考 auto_aim Camera::processImage() 方法
// ============================================================
bool processCameraFrame(unsigned char* pData, MV_FRAME_OUT_INFO_EX& stImageInfo, cv::Mat& outputImage) {
    switch (stImageInfo.enPixelType) {
        case PixelType_Gvsp_BayerGB8:
        case PixelType_Gvsp_BayerRG8:
        case PixelType_Gvsp_BayerGR8:
        case PixelType_Gvsp_BayerBG8: {
            cv::Mat rawImg(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
            int conversionCode;
            switch (stImageInfo.enPixelType) {
                case PixelType_Gvsp_BayerGB8: conversionCode = cv::COLOR_BayerGB2BGR; break;
                case PixelType_Gvsp_BayerRG8: conversionCode = cv::COLOR_BayerRG2BGR; break;
                case PixelType_Gvsp_BayerGR8: conversionCode = cv::COLOR_BayerGR2BGR; break;
                case PixelType_Gvsp_BayerBG8: conversionCode = cv::COLOR_BayerBG2BGR; break;
                default:                      conversionCode = cv::COLOR_BayerGB2BGR; break;
            }
            cv::Mat bgrImg;
            cv::cvtColor(rawImg, bgrImg, conversionCode);
            // Bayer 格式下需要交换 R/B 通道
            std::vector<cv::Mat> channels(3);
            cv::split(bgrImg, channels);
            std::swap(channels[0], channels[2]);
            cv::merge(channels, bgrImg);
            outputImage = bgrImg;
            return true;
        }
        case PixelType_Gvsp_BGR8_Packed: {
            outputImage = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData).clone();
            return true;
        }
        case PixelType_Gvsp_RGB8_Packed: {
            cv::Mat rgbImg(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);
            cv::cvtColor(rgbImg, outputImage, cv::COLOR_RGB2BGR);
            return true;
        }
        case PixelType_Gvsp_Mono8: {
            cv::Mat grayImg(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
            cv::cvtColor(grayImg, outputImage, cv::COLOR_GRAY2BGR);
            return true;
        }
        default:
            cerr << "[ERROR] Unsupported pixel format: " << stImageInfo.enPixelType << endl;
            return false;
    }
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char** argv) {
    // 注册信号处理
    signal(SIGINT, signalHandler);

    // -------------------- 解析命令行参数 --------------------
    string model_path   = "Model/0526.onnx";
    string device_name  = "CPU";
    int    detect_color = -1;   // -1: 自动（红蓝都检测）, 0: 只检测红色, 1: 只检测蓝色
    int    camera_index = 0;    // USB 相机索引（仅 USB 模式）
    string cam_ip       = "";   // GigE 相机 IP
    string pc_ip        = "";   // 本机 IP

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)
            model_path = argv[++i];
        else if (arg == "--device" && i + 1 < argc)
            device_name = argv[++i];
        else if (arg == "--color" && i + 1 < argc)
            detect_color = atoi(argv[++i]);
        else if (arg == "--usb" && i + 1 < argc)
            camera_index = atoi(argv[++i]);
        else if (arg == "--gige-ip" && i + 1 < argc)
            cam_ip = argv[++i];
        else if (arg == "--pc-ip" && i + 1 < argc)
            pc_ip = argv[++i];
        else if (arg == "--help") {
            cout << "Usage: armor_detection [options]\n"
                 << "  --model <path>      ONNX model path (default: Model/0526.onnx)\n"
                 << "  --device <name>     OpenVINO device: CPU, GPU, AUTO (default: CPU)\n"
                 << "  --color <0|1>       Detect color: 0=RED, 1=BLUE, -1=BOTH (default: -1)\n"
                 << "  --usb <index>       USB camera index (default: 0)\n"
                 << "  --gige-ip <ip>      GigE camera IP (e.g., 192.168.1.100)\n"
                 << "  --pc-ip <ip>        PC IP for GigE (e.g., 192.168.1.10)\n"
                 << "  --help              Show this help\n";
            return 0;
        }
    }

    cout << "==========================================" << endl;
    cout << "   Armor Detection with OpenVINO" << endl;
    cout << "==========================================" << endl;
    cout << "Model:      " << model_path << endl;
    cout << "Device:     " << device_name << endl;
    cout << "DetectColor:" << (detect_color == -1 ? "BOTH" : (detect_color == 0 ? "RED" : "BLUE")) << endl;

    // -------------------- Step 1: 模型转换（ONNX -> IR） --------------------
    // 检查模型文件是否存在
    if (FILE* f = fopen(model_path.c_str(), "r")) {
        fclose(f);
    } else {
        cerr << "[ERROR] Model file not found: " << model_path << endl;
        return -1;
    }

    // 检查是否已经有同名的 .xml 文件（避免重复转换）
    string base_path = model_path;
    size_t dot_pos = base_path.rfind(".onnx");
    if (dot_pos != string::npos) base_path = base_path.substr(0, dot_pos);
    string xml_path_str = base_path + ".xml";
    string bin_path_str = base_path + ".bin";

    // 检查 .xml 和 .bin 是否都已存在
    bool need_convert = true;
    FILE* f_xml = fopen(xml_path_str.c_str(), "r");
    FILE* f_bin = fopen(bin_path_str.c_str(), "r");
    if (f_xml && f_bin) {
        need_convert = false;
        cout << "[INFO] IR files already exist, skipping conversion." << endl;
    }
    if (f_xml) fclose(f_xml);
    if (f_bin) fclose(f_bin);

    if (need_convert) {
        auto [xml_path, bin_path] = convertOnnxToIR(model_path);
        xml_path_str = xml_path;
        bin_path_str = bin_path;
        cout << "[INFO] Model converted to IR format successfully!" << endl;
    }

    // -------------------- Step 2: 初始化推理器 --------------------
    // 使用 OpenvinoInfer 的第一个构造函数
    // （参考 OpenvinoInfer.cpp 中的实现：BGR输入 -> RGB -> 归一化 -> NCHW）
    OpenvinoInfer infer(xml_path_str, bin_path_str, device_name);
    cout << "[INFO] Inference model loaded successfully!" << endl;

    // -------------------- Step 3: 初始化海康相机（参考 auto_aim 的 Camera 类）--------------------
    void* handle = nullptr;
    bool camera_ready = false;

    // 初始化 SDK
    int nRet = MV_CC_Initialize();
    if (MV_OK != nRet) {
        cerr << "[ERROR] MV SDK Initialize fail! nRet: 0x" << hex << nRet << dec << endl;
        return -1;
    }

    // 枚举设备，选择合适的连接方式
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    bool use_gige = !cam_ip.empty();

    if (use_gige) {
        // === GigE 相机连接（参考 Camera::tryConnectGigE()） ===
        cout << "[INFO] Connecting GigE camera at " << cam_ip << " ..." << endl;

        MV_CC_DEVICE_INFO stDevInfo;
        MV_GIGE_DEVICE_INFO stGigEDev;
        memset(&stDevInfo, 0, sizeof(MV_CC_DEVICE_INFO));
        memset(&stGigEDev, 0, sizeof(MV_GIGE_DEVICE_INFO));

        // 解析 IP 地址
        auto parseIp = [](const string& ip, unsigned int& parsedIp) {
            int parts[4];
            sscanf(ip.c_str(), "%d.%d.%d.%d", &parts[0], &parts[1], &parts[2], &parts[3]);
            parsedIp = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3];
        };
        parseIp(cam_ip, stGigEDev.nCurrentIp);
        parseIp(pc_ip, stGigEDev.nNetExport);

        stDevInfo.nTLayerType = MV_GIGE_DEVICE;
        stDevInfo.SpecialInfo.stGigEInfo = stGigEDev;

        // 创建句柄
        nRet = MV_CC_CreateHandle(&handle, &stDevInfo);
        if (MV_OK != nRet) {
            cerr << "[ERROR] Create GigE Handle fail! nRet: 0x" << hex << nRet << dec << endl;
            MV_CC_Finalize();
            return -1;
        }

        // 打开设备
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet) {
            cerr << "[ERROR] Open GigE device fail! nRet: 0x" << hex << nRet << dec << endl;
            MV_CC_DestroyHandle(handle);
            MV_CC_Finalize();
            return -1;
        }

        // 设置最佳数据包大小
        int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
        if (nPacketSize > 0) {
            MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
        }

        camera_ready = true;
        cout << "[INFO] GigE camera connected!" << endl;

    } else {
        // === USB 相机连接（参考 Camera::tryConnectUSB()） ===
        cout << "[INFO] Enumerating USB cameras..." << endl;

        nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet || stDeviceList.nDeviceNum == 0) {
            cerr << "[ERROR] No USB camera found!" << endl;
            MV_CC_Finalize();
            return -1;
        }

        cout << "[INFO] Found " << stDeviceList.nDeviceNum << " USB camera(s)." << endl;
        for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++) {
            MV_CC_DEVICE_INFO* pInfo = stDeviceList.pDeviceInfo[i];
            if (pInfo->nTLayerType == MV_USB_DEVICE) {
                cout << "  [" << i << "] "
                     << pInfo->SpecialInfo.stUsb3VInfo.chModelName
                     << " (SN: " << pInfo->SpecialInfo.stUsb3VInfo.chSerialNumber << ")" << endl;
            }
        }

        if (camera_index >= (int)stDeviceList.nDeviceNum) {
            cerr << "[ERROR] Camera index " << camera_index << " out of range!" << endl;
            MV_CC_Finalize();
            return -1;
        }

        // 创建句柄
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[camera_index]);
        if (MV_OK != nRet) {
            cerr << "[ERROR] Create USB Handle fail! nRet: 0x" << hex << nRet << dec << endl;
            MV_CC_Finalize();
            return -1;
        }

        // 打开设备
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet) {
            cerr << "[ERROR] Open USB device fail! nRet: 0x" << hex << nRet << dec << endl;
            MV_CC_DestroyHandle(handle);
            MV_CC_Finalize();
            return -1;
        }

        camera_ready = true;
        cout << "[INFO] USB camera connected!" << endl;
    }

    // -------------------- 设置相机参数（参考 Camera::initCameraCommonParams()）--------------------
    if (camera_ready) {
        // 禁用自动曝光
        MV_CC_SetEnumValue(handle, "ExposureAuto", 0);
        // 禁用自动增益
        MV_CC_SetEnumValue(handle, "GainAuto", 0);
        // 禁用自动白平衡
        MV_CC_SetEnumValue(handle, "BalanceWhiteAuto", 0);
        // 设置曝光时间（微秒）—— 低曝光适合装甲板检测
        MV_CC_SetFloatValue(handle, "ExposureTime", 5000.0f);
        // 设置增益
        MV_CC_SetFloatValue(handle, "Gain", 16.0f);

        cout << "[INFO] Camera parameters set." << endl;
    }

    // -------------------- Step 4: 开始取流 --------------------
    cout << "[INFO] Starting grabbing..." << endl;
    nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet) {
        cerr << "[ERROR] Start grabbing fail! nRet: 0x" << hex << nRet << dec << endl;
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
        MV_CC_Finalize();
        return -1;
    }
    cout << "[INFO] Grabbing started! Press 'q' or ESC to quit." << endl;

    // 获取 PayloadSize
    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
    unsigned int nPayloadSize = (MV_OK == nRet) ? stParam.nCurValue : 1280 * 1024 * 3;
    unsigned char* pFrameBuffer = new unsigned char[nPayloadSize];

    // -------------------- FPS 统计 --------------------
    auto fps_last_time = chrono::steady_clock::now();
    int fps_frame_count = 0;
    float current_fps = 0.0f;

    // -------------------- Step 5: 主循环 --------------------
    while (!g_bExit) {
        MV_FRAME_OUT_INFO_EX stImageInfo;
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));

        // 采集一帧图像
        nRet = MV_CC_GetOneFrameTimeout(handle, pFrameBuffer, nPayloadSize, &stImageInfo, 1000);
        if (nRet != MV_OK) {
            // 超时，继续
            continue;
        }

        // 转换为 OpenCV Mat
        cv::Mat frame;

        if (!processCameraFrame(pFrameBuffer, stImageInfo, frame)) {
            continue;
        }
        
        if (frame.empty()) {
            continue;
        }

        // =========== 推理（缩放到640x640后推理，坐标映射回原图再绘制）===========
        // 使用 OpenvinoInfer::infer() 方法（参考 OpenvinoInfer.cpp 实现）
        // detect_color: -1 表示红蓝都检测
        int infer_color;
        if (detect_color == -1) {
            // 不过滤颜色：给一个 {0,1} 之外的值，内部两个 if 都不触发
            infer_color = 2;
        } else if (detect_color == 0) {
            // 检测红色：给 detect_color=1 表示跳过蓝色 (color_id.x==0)
            infer_color = 1;
        } else {
            // 检测蓝色：给 detect_color=0 表示跳过红色 (color_id.x==1)
            infer_color = 0;
        }

        // 1. 缩放到 640x640 进行推理（模型输入要求 640x640）
        cv::Mat infer_frame;
        cv::resize(frame, infer_frame, cv::Size(640, 640));

        auto infer_start = chrono::steady_clock::now();
        infer.infer(infer_frame, infer_color);
        auto infer_end = chrono::steady_clock::now();
        double infer_time_ms = chrono::duration_cast<chrono::microseconds>(infer_end - infer_start).count() / 1000.0;

        // 2. 将检测结果的坐标从 640x640 映射回原图尺寸
        float scale_x = (float)frame.cols / 640.0f;
        float scale_y = (float)frame.rows / 640.0f;
        int img_w = frame.cols, img_h = frame.rows;
        vector<Object> display_objects = infer.tmp_objects;
        for (auto& obj : display_objects) {
            obj.rect.x      = (int)(obj.rect.x * scale_x);
            obj.rect.y      = (int)(obj.rect.y * scale_y);
            obj.rect.width  = (int)(obj.rect.width * scale_x);
            obj.rect.height = (int)(obj.rect.height * scale_y);
            for (int i = 0; i < 8; i += 2) {
                obj.landmarks[i]   *= scale_x;
                obj.landmarks[i+1] *= scale_y;
            }
            obj.length *= scale_x;
            obj.width  *= scale_y;

            // 钳位 rect 到图像范围内，防止绘制时越界
            obj.rect.x = std::max(0.0f, obj.rect.x);
            obj.rect.y = std::max(0.0f, obj.rect.y);
            obj.rect.width  = std::min(obj.rect.width,  img_w - obj.rect.x);
            obj.rect.height = std::min(obj.rect.height, img_h - obj.rect.y);
        }

        // 3. 绘制检测结果（在原图尺寸上绘制，不是缩略图）
        cv::Mat display = frame.clone();
        drawDetectionResults(display, display_objects);

        // =========== 显示信息 ===========
        // 计算 FPS
        fps_frame_count++;
        auto now = chrono::steady_clock::now();
        double elapsed_ms = chrono::duration_cast<chrono::milliseconds>(now - fps_last_time).count();
        if (elapsed_ms >= 1000.0) {
            current_fps = fps_frame_count * 1000.0f / elapsed_ms;
            fps_frame_count = 0;
            fps_last_time = now;
        }

        // 左上角显示信息
        string info_text = cv::format("FPS: %.1f | Infer: %.1f ms | Objects: %zu",
                                      current_fps, infer_time_ms, infer.tmp_objects.size());
        cv::putText(display, info_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        string color_text = cv::format("Detect: %s | Device: %s",
                                       detect_color == -1 ? "BOTH" : (detect_color == 0 ? "RED" : "BLUE"),
                                       device_name.c_str());
        cv::putText(display, color_text, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        // 显示图像分辨率
        string res_text = cv::format("Resolution: %dx%d", frame.cols, frame.rows);
        cv::putText(display, res_text, cv::Point(10, 90),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        // =========== 显示图像 ===========
        cv::imshow("Armor Armor Detection", display);

        // 按键检测
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {  // 27 = ESC
            break;
        }
    }

    // -------------------- 清理资源 --------------------
    cout << "[INFO] Cleaning up..." << endl;

    delete[] pFrameBuffer;

    if (handle != nullptr) {
        MV_CC_StopGrabbing(handle);
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
    }
    MV_CC_Finalize();

    cv::destroyAllWindows();

    // 清理生成的 IR 临时文件（可选）
    // 如果希望保留 IR 文件以便下次快速启动，可以注释掉删除逻辑
    // remove(xml_path_str.c_str());
    // remove(bin_path_str.c_str());

    cout << "[INFO] Program exited." << endl;
    return 0;
}
