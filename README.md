<h2>OnnxNetKeypoint</h2>

VideoMatrix OnnxNetKeypoint project.

<h2>Run</h2>

Clone project and open by Onnxnet_project.sln

<h2>Requirements</h2>

1. [OpenCV](https://opencv.org/ "OpenCV")
2. [CUDA >= 11.3](https://developer.nvidia.com/cuda-downloads/ "CUDA")
3. [Cudnn](https://developer.nvidia.com/cudnn/ "Cudnn")

<h2>Structure</h2>
The project contains the following modules as a mono-repository:

<h3>Projects</h3>

1. <b>onnxnet</b> - library project
2. <b>test</b> - exe project (with main - run example)

Class OnnxNet have a constructer like: 

onnxnet::OnnxNet model(model path, size of input, heatmap size);

In test main.cpp

int main() {
	
	onnxnet::OnnxNet model("body_deeppose_res50_coco_256x192.onnx", cv::Size(192, 256), cv::Size( 1, 1 ));
	//onnxnet::OnnxNet model("face_res50_coco_wholebody_face_256x256.onnx", cv::Size(256, 256), cv::Size( 64, 64 ));
	//onnxnet::OnnxNet model("wholebody_res50_coco_wholebody_256x192.onnx" , cv::Size(192, 256), cv::Size( 48, 64 ));
	//onnxnet::OnnxNet model("hand_res50_coco_wholebody_hand_256x256.onnx", cv::Size(256, 256), cv::Size( 64, 64 ));

	std::vector<cv::Mat> images;
	images.push_back(cv::imread("./test/test1.jpg"));
	images.push_back(cv::imread("./test/test2.jpg"));
	images.push_back(cv::imread("./test/test3.jpg"));
	images.push_back(cv::imread("./test/test4.png"));
	images.push_back(cv::imread("./test/test5.jpg"));

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


