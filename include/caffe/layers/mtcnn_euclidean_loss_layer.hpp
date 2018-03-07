#ifndef CAFFE_MTCNN_EUCLIDEAN_LOSS_LAYER_HPP_
#define CAFFE_MTCNN_EUCLIDEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
	
template <typename Dtype>
class MTCNNEuclideanLossLayer : public LossLayer<Dtype> {
public:
	explicit MTCNNEuclideanLossLayer(const LayerParameter& param)
		: LossLayer<Dtype>(param), diff_() {}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline int ExactNumBottomBlobs() const { return -1; }
	virtual inline int MinBottomBlobs() const { return 2; }
	virtual inline int MaxBottomBlobs() const { return 3; }
	virtual inline const char* type() const { return "MTCNNEuclideanLoss"; }

	/**
	* Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
	* to both inputs -- override to return true and always allow force_backward.
	*/
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return true;
	}
protected:
	/// @copydoc EuclideanLossLayer
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> diff_;

};

	

} // namespace caffe


#endif	//CAFFE_MTCNN_EUCLIDEAN_LOSS_LAYER_HPP_