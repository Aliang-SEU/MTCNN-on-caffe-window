#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include <cstdint>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/rng.hpp"
#include <opencv2/opencv.hpp>
#include "caffe/util/io.hpp"

using std::string;
using std::vector;

DEFINE_string(backend, "lmdb", 
	"The backend for storing the result");
DEFINE_bool(shuffle, false,
	"Randomly shuffle the order of images and their labels");
DEFINE_int32(img_size, 12,
	"Image Size");

void MtcnnDataGenerate(const string& input_file_name) {
	std::ifstream input_file(input_file_name, std::ios::in);


}
bool ReadImageToMTCNNDatum(const string& filename, const vector<float>& bbox, caffe::MTCNNDatum & mtcnn_datum, bool is_color)
{
	int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	else {
		mtcnn_datum.clear_rois();
		CHECK_EQ(bbox.size(), 5) << "Size Error!";
		auto dt = mtcnn_datum.mutable_datum();
		CVMatToDatum(cv_img_origin, dt);
		dt->set_label(int(bbox[0]));
		auto rois = mtcnn_datum.rois();
		rois.set_xmin(bbox[1]);
		rois.set_ymin(bbox[2]);
		rois.set_xmax(bbox[3]);
		rois.set_ymax(bbox[4]);
	}
	return true;
}
void convert_data(const string& input_file_name, const string& output_folder, const string& root_folder, const string& db_backend) {
	//打开文件
	//std::ifstream input_file(input_file_name, std::ios::in | std::ios::binary);
	//CHECK(input_file) << "Error to open file" << input_file_name;

	const string& img_size = caffe::format_int(FLAGS_img_size);
	std::ifstream input_file(root_folder + "/" + img_size + "/" + input_file_name, std::ios::in);
	CHECK(input_file) << "Fail to open file " << input_file_name;

	vector<std::pair<string, vector<float>> > lines;	//用于保存每一行的数据
	string line, file_name;

	while(std::getline(input_file, line)){
		float label;
		std::istringstream iss(line);
		iss >> file_name;
		std::vector<float> labels;
		while(iss >> label){
			labels.push_back(label);
		}
		CHECK_EQ(labels.size(), 5) << "mtcnn data size error" << file_name; 
		lines.push_back(std::make_pair(file_name, labels));
	}

	if (FLAGS_shuffle) {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		caffe::shuffle(lines.begin(), lines.end());
	}

	LOG(INFO) << "A total of " << lines.size() << " images.";

	boost::scoped_ptr<caffe::db::DB> db(caffe::db::GetDB(db_backend));//新建一个数据库
	db->Open(output_folder + "/mtcnn_train_" + db_backend, caffe::db::NEW);
	boost::scoped_ptr<caffe::db::Transaction> txn(db->NewTransaction());

	int count = 0;

	caffe::MTCNNDatum mtcnndatum;

	LOG(INFO) << "Reading train data...";
	for (int line_id = 0; line_id < lines.size(); ++line_id) {
		if(line_id % 1000 ==0)
			LOG(INFO) << "Loading image " << line_id + 1;

		file_name = root_folder + "/" + lines[line_id].first;
		bool status = ReadImageToMTCNNDatum(file_name, lines[line_id].second, mtcnndatum, true);
		if (status == false) continue;

		string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;
		string out;
		CHECK(mtcnndatum.SerializeToString(&out));
		txn->Put(key_str, out);

		if (++count % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(INFO) << "Processed " << count << " files.";
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
		LOG(INFO) << "Processed " << count << " files.";
	}

}

int main(int argc, char ** argv) {

	FLAGS_alsologtostderr = 1;

	gflags::SetUsageMessage("This script converts the cifar-10 dataset to\n"
		"the lmdb format used by Caffe to load data.\n"
		"Usage:\n"
		"    mtcnn_data_convert [FLAGS] input_file output_folder root_folder\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);	//解析命令行的数据

	const string& db_backend = FLAGS_backend;

	if (argc != 4) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "cifar-10_data");
	}
	else {
		google::InitGoogleLogging(argv[0]);
		convert_data(string(argv[1]), string(argv[2]), string(argv[3]), db_backend);
	}

	return 0;
}


