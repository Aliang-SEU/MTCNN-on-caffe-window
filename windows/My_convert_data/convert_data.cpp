#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include <cstdint>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
using std::string;


DEFINE_string(backend, "lmdb", "The backend for storing the result");

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;
const int rows = 32, cols = 32;

void read_image(std::ifstream* file, char* image_data, int* label){
	char label_char;
	file->read(&label_char, 1);
	*label = label_char;
	file->read(image_data, kCIFARImageNBytes);
	return;
}
void convert_data(const string& input_folder, const string& output_folder, const string& db_backend){
	//打开文件
	//std::ifstream input_file(input_file_name, std::ios::in | std::ios::binary);
	//CHECK(input_file) << "Error to open file" << input_file_name;
	


	boost::scoped_ptr<caffe::db::DB> db(caffe::db::GetDB(db_backend));//新建一个数据库
	db->Open(output_folder + "/cifar10-data-" + db_backend, caffe::db::NEW);
	boost::scoped_ptr<caffe::db::Transaction> txn(db->NewTransaction());

	int label;
	char* piexls = new char[kCIFARImageNBytes];
	int count = 0;
	string value;

	caffe::Datum datum;
	caffe::MTCNNDatum mtcnndatum;

	datum.set_channels(3);
	datum.set_height(rows);
	datum.set_width(cols);
	
	LOG(INFO) << "Writing train data...";
	for(int i = 0 ; i < kCIFARTrainBatches; ++i){
		LOG(INFO) << "Training Batch " << i + 1;
		string batch_file_name = input_folder + "/data_batch_" + caffe::format_int(i + 1) + ".bin";
		std::ifstream input_file(batch_file_name.c_str(), std::ios::in | std::ios::binary);
		CHECK(input_file) << "Error to open file" << batch_file_name;
		for(int j = 0; j < kCIFARBatchSize; ++j){
			read_image(&input_file, piexls, &label);
			count++;
			datum.set_data(piexls, kCIFARImageNBytes);
			datum.set_label(label);
			string out;
			CHECK(datum.SerializeToString(&out));
			txn->Put(caffe::format_int(i * kCIFARBatchSize + j, 5), out);
		}
	}
	txn->Commit();
	db->Close();

	LOG(INFO) << "Writing test data...";
	boost::scoped_ptr<caffe::db::DB> test_db(caffe::db::GetDB(db_backend));
	test_db->Open(output_folder + "/cifar-10-test-" + db_backend, caffe::db::NEW);
	txn.reset(test_db->NewTransaction());
	
	string test_file_name = input_folder + "/test_batch.bin";
	std::ifstream test_file(test_file_name, std::ios::in | std::ios::binary);
	CHECK(test_file) << "Error to open file " << test_file_name;

	for(int i = 0; i < kCIFARBatchSize; ++i){
		read_image(&test_file, piexls, &label);
		datum.set_label(label);
		datum.set_data(piexls, kCIFARImageNBytes);
		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(caffe::format_int(i, 5), out);
	}
	txn->Commit();
	test_db->Close();

}

int main(int argc,char ** argv){

	FLAGS_alsologtostderr = 1;

	gflags::SetUsageMessage("This script converts the cifar-10 dataset to\n"
		"the lmdb format used by Caffe to load data.\n"
		"Usage:\n"
		"    convert_data [FLAGS] input_file "
		"output_db_file\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);	//解析命令行的数据

	const string& db_backend = FLAGS_backend;

	if(argc != 3){
		gflags::ShowUsageWithFlagsRestrict(argv[0], "cifar-10_data");
	}else{
		google::InitGoogleLogging(argv[0]);
		convert_data(string(argv[1]), string(argv[2]), db_backend);
	}

	return 0;
}


