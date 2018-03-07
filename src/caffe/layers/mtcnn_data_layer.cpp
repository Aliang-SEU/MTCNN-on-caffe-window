#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/mtcnn_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "opencv2/highgui.hpp"

namespace caffe {

	template <typename Dtype>
	MTCNNDataLayer<Dtype>::MTCNNDataLayer(const LayerParameter& param)
		: BasePrefetchingDataLayer<Dtype>(param),
		offset_() {
		db_.reset(db::GetDB(param.data_param().backend()));
		db_->Open(param.data_param().source(), db::READ);
		cursor_.reset(db_->NewCursor());
	}

	template <typename Dtype>
	MTCNNDataLayer<Dtype>::~MTCNNDataLayer() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void MTCNNDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int batch_size = this->layer_param_.data_param().batch_size();
		// Read a data point, and use it to initialize the top blob.
		MTCNNDatum datum;
		datum.ParseFromString(cursor_->value());

		// Use data_transformer to infer the expected blob shape from datum.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(datum.datum());
		this->transformed_data_.Reshape(top_shape);
		// Reshape top[0] and prefetch_data according to the batch_size.
		top_shape[0] = batch_size;
		top[0]->Reshape(top_shape);
		for (int i = 0; i < this->prefetch_.size(); ++i) {
			this->prefetch_[i]->data_.Reshape(top_shape);
		}
		LOG_IF(INFO, Caffe::root_solver())
			<< "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();
		// label
		if (this->output_labels_) {
			vector<int> label_shape(1);
			label_shape[0] = batch_size;
			top[1]->Reshape(label_shape);
			for (int i = 0; i < this->prefetch_.size(); ++i) {
				this->prefetch_[i]->label_.Reshape(label_shape);
			}
		}
		// roi
		if (this->output_roi_) {
			vector<int> roi_shape(2);
			roi_shape[0] = batch_size;
			roi_shape[1] = 4;
			top[2]->Reshape(roi_shape);
			for (int i = 0; i < this->prefetch_.size(); ++i) {
				this->prefetch_[i]->roi_.Reshape(roi_shape);
			}
			// pts
			if (this->output_pts_) {
				vector<int> pts_shape(2);
				pts_shape[0] = batch_size;
				pts_shape[1] = 10;			//默认为5个坐标点 即 10 个points
				top[3]->Reshape(roi_shape);
				for (int i = 0; i < this->prefetch_.size(); ++i) {
					this->prefetch_[i]->roi_.Reshape(roi_shape);
				}
			}
		}
	}

		template <typename Dtype>
		bool MTCNNDataLayer<Dtype>::Skip() {
			int size = Caffe::solver_count();
			int rank = Caffe::solver_rank();
			bool keep = (offset_ % size) == rank ||
				// In test mode, only rank 0 runs, so avoid skipping
				this->layer_param_.phase() == TEST;
			return !keep;
		}

		template<typename Dtype>
		void MTCNNDataLayer<Dtype>::Next() {
			cursor_->Next();
			if (!cursor_->valid()) {
				//LOG_IF(INFO, Caffe::root_solver())
					//<< "Restarting data prefetching from start.";
				cursor_->SeekToFirst();
			}
			offset_++;
		}

		// This function is called on prefetch thread
		template<typename Dtype>
		void MTCNNDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
			CPUTimer batch_timer;
			batch_timer.Start();
			double read_time = 0;
			double trans_time = 0;
			CPUTimer timer;
			CHECK(batch->data_.count());
			CHECK(this->transformed_data_.count());
			const int batch_size = this->layer_param_.data_param().batch_size();

			MTCNNDatum datum;
			for (int item_id = 0; item_id < batch_size; ++item_id) {
				timer.Start();
				while (Skip()) {
					Next();
				}
				datum.ParseFromString(cursor_->value());	//数据库内读入一条数据
				const Datum& data = datum.datum();
				read_time += timer.MicroSeconds();
				//cv::Mat cv_img = DatumToCVMat(data);
				//cv::imshow("1", cv_img);
				//cv::waitKey(0);
				//DecodeDatumToCVMat(data, true);
				if (item_id == 0) {
					// Reshape according to the first datum of each batch
					// on single input batches allows for inputs of varying dimension.
					// Use data_transformer to infer the expected blob shape from datum.
					vector<int> top_shape = this->data_transformer_->InferBlobShape(data);
					this->transformed_data_.Reshape(top_shape);
					// Reshape batch according to the batch_size.
					top_shape[0] = batch_size;
					batch->data_.Reshape(top_shape);
				}

				// Apply data transformations (mirror, scale, crop...)
				timer.Start();
				int offset = batch->data_.offset(item_id);
				Dtype* top_data = batch->data_.mutable_cpu_data();
				this->transformed_data_.set_cpu_data(top_data + offset);
				this->data_transformer_->Transform(data, &(this->transformed_data_));
				// Copy label.
				if (this->output_labels_) {
					Dtype* top_label = batch->label_.mutable_cpu_data();
					top_label[item_id] = data.label();
				}
				// Copy rois.
				if (this->output_roi_) {
					Dtype* top_roi = batch->roi_.mutable_cpu_data();
					top_roi[item_id * 4 + 0] = datum.rois().xmin();
					top_roi[item_id * 4 + 1] = datum.rois().ymin();
					top_roi[item_id * 4 + 2] = datum.rois().xmax();
					top_roi[item_id * 4 + 3] = datum.rois().ymax();
				}
				// Copy pts
				if (this->output_pts_) {
					Dtype* top_pts = batch->pts_.mutable_cpu_data();
					//Dtype& bottom_pts = datum.pts();
					int pts_size = datum.pts_size();	//默认应该为10
					CHECK_EQ(pts_size, 10) << "pts size error";
					for (int i = 0; i < pts_size; ++i) {
						top_pts[item_id * pts_size + i] = datum.pts(i);
					}
				}
				trans_time += timer.MicroSeconds();
				Next();	//数据库指针后移
			}
			timer.Stop();
			batch_timer.Stop();
			//DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
			//DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
			//DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
		}
	INSTANTIATE_CLASS(MTCNNDataLayer);
	REGISTER_LAYER_CLASS(MTCNNData);
}  // namespace caffe
