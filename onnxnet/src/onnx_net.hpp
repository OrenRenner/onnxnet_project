#pragma once
#include<iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

namespace onnxnet {
	class OnnxNet {
	public:
		OnnxNet(const std::string& model, const std::string device = "GPU");
		~OnnxNet();

		void* calculate(const cv::Mat& mat, size_t& count);
	private:
		cv::dnn::Net net;
		cv::Size input_size;
	};
};