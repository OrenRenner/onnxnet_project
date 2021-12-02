#include "onnx_net.hpp"

onnxnet::OnnxNet::OnnxNet(const std::string& model, const std::string device)
{
    this->net = cv::dnn::readNetFromONNX(model);
    this->input_size = cv::Size(192, 256);
    if (device == "GPU") {
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
}

onnxnet::OnnxNet::~OnnxNet()
{
}

void* onnxnet::OnnxNet::calculate(const cv::Mat& mat, size_t& count)
{
    try {
        cv::Mat resized;
        cv::resize(mat, resized, this->input_size, 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(resized, resized,
            cv::ColorConversionCodes::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0 / 255);
        cv::subtract(resized, cv::Scalar(0.485f, 0.456f, 0.406f), resized, cv::noArray(), -1);
        cv::divide(resized, cv::Scalar(0.229f, 0.224f, 0.225f), resized, 1, -1);

        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(resized, preprocessedImage);

        this->net.setInput(preprocessedImage);
        cv::Mat outputs = this->net.forward();

        int cout_point = 1;
        for (int i = 0; i < outputs.size.dims(); i++) cout_point *= outputs.size[i];

        //size_t inputTensorSize = vectorProduct(this->inputDims);
        std::vector<float> inputTensorValues(cout_point);
        inputTensorValues.assign(outputs.begin<float>(),
            outputs.end<float>());

        std::vector<cv::Point> points;
        int x_coor = 0, y_coor = 0;
        for (int i = 0; i < cout_point; i++) {
            if (i % 2 == 0) x_coor = round(inputTensorValues[i] * mat.cols);
            else y_coor = round(inputTensorValues[i] * mat.rows);

            if (y_coor == 0 && x_coor == 0) break;

            if (y_coor != 0 && x_coor != 0) {
                cv::Point tmp(x_coor, y_coor);
                points.push_back(tmp);
                x_coor = 0; y_coor = 0;
            }
        }

        cv::Point* result = new cv::Point[points.size()];
        for (int i = 0; i < points.size(); i++) result[i] = points[i];
        count = points.size();

        return static_cast<void*>(result);
    }
    catch (std::exception) {
        return nullptr;
    }
}
