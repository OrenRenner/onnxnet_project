#include"../onnxnet/src/onnx_net.hpp"

int main() {
	onnxnet::OnnxNet model("body_deeppose_res50_coco_256x192.onnx");

	std::vector<cv::Mat> images;
	images.push_back(cv::imread("./test/test1.jpg"));
	images.push_back(cv::imread("./test/test2.jpg"));
	images.push_back(cv::imread("./test/test3.jpg"));
	images.push_back(cv::imread("./test/test4.png"));

	for (int i = 0; i < images.size(); i++) {
		cv::Mat tmp_image = images[i];

		size_t count = 0;
		cv::Point* result = static_cast<cv::Point*>(model.calculate(tmp_image, count));

		for (int i = 0; i < count; i++) {
			cv::circle(tmp_image, result[i], 2, cv::Scalar(255, 0, 0));
		}

		// Show result
		cv::namedWindow("image", 0);
		cv::imshow("image", tmp_image);
		cv::waitKey(0);
	}

	return 0;
}