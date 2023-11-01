#include <torch/extension.h>
#include <vector>

#ifdef VERSION_GE_1_3
#define DATA_PTR data_ptr
#else
#define DATA_PTR data
#endif

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void vote_nms_cuda(at::Tensor pcds_fg, at::Tensor pcds_center, float dist_thresh, float vote_thresh);
void vote_nms_fast_cuda(at::Tensor pcds_fg, at::Tensor pcds_center, at::Tensor matching_mat, at::Tensor matching_mat_vote, at::Tensor remv, at::Tensor keep, float dist_thresh, float vote_thresh);

void voxel_sum_cuda(at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size);
void voxel_query_cuda(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_in, at::Tensor voxel_in_size, at::Tensor voxel_in_stride);


void vote_nms_gpu(at::Tensor pcds_fg, at::Tensor pcds_center, float dist_thresh, float vote_thresh){
  CHECK_INPUT(pcds_fg);
  CHECK_INPUT(pcds_center);

  vote_nms_cuda(pcds_fg, pcds_center, dist_thresh, vote_thresh);
}

void vote_nms_fast_gpu(at::Tensor pcds_fg, at::Tensor pcds_center, at::Tensor matching_mat, at::Tensor matching_mat_vote, at::Tensor remv, at::Tensor keep, float dist_thresh, float vote_thresh){
  CHECK_INPUT(pcds_fg);
  CHECK_INPUT(pcds_center);
  CHECK_INPUT(matching_mat);
  CHECK_INPUT(matching_mat_vote);
  CHECK_INPUT(remv);
  CHECK_INPUT(keep);

  vote_nms_fast_cuda(pcds_fg, pcds_center, matching_mat, matching_mat_vote, remv, keep, dist_thresh, vote_thresh);
}


void voxel_sum_gpu(at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size)
{
  CHECK_INPUT(pcds_ind);
  CHECK_INPUT(voxel_out);

  CHECK_INPUT(voxel_out_size);
  CHECK_INPUT(voxel_out_stride);
  CHECK_INPUT(output_size);

  voxel_sum_cuda(pcds_ind, voxel_out, voxel_out_size, voxel_out_stride, output_size);
}

void voxel_query_gpu(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_in, at::Tensor voxel_in_size, at::Tensor voxel_in_stride)
{
  CHECK_INPUT(pcds_feat);
  CHECK_INPUT(pcds_ind);
  CHECK_INPUT(voxel_in);

  CHECK_INPUT(voxel_in_size);
  CHECK_INPUT(voxel_in_stride);

  voxel_query_cuda(pcds_feat, pcds_ind, voxel_in, voxel_in_size, voxel_in_stride);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vote_nms_gpu", &vote_nms_gpu, "vote nms on GPU");
  m.def("vote_nms_fast_gpu", &vote_nms_fast_gpu, "fast vote nms on GPU");
  m.def("voxel_sum_gpu", &voxel_sum_gpu, "voxel sum gpu function");
  m.def("voxel_query_gpu", &voxel_query_gpu, "voxel query gpu function");
}