// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/xpu/multiclass_nms_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void MulticlassNmsCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* boxes = param.bboxes;
  auto* scores = param.scores;
  auto* outs = param.out;
  auto* out_index = param.index;
  bool return_index = param.index ? true : false;
  int n = boxes->dims()[0];
  int box_num = boxes->dims()[1];
  int class_num = scores->dims()[1];
  int out_dim = boxes->dims()[2] + 2; // 4 + 2
  CHECK(class_num <= 80)
      << "xpu MulticlassNms only support class_num <= 80 which is "
      << class_num;

  outs->Resize({n, box_num, out_dim});
  if (return_index) {
      out_index->Resize({n, box_num});
  }

  std::vector<size_t> batch_starts = {0};
  int r = xdnn::multiclass_nms2<float, int>(
      ctx.GetRawContext(), /* context */
      boxes->data<float>(),
      scores->data<float>(),
      outs->mutable_data<float>(TARGET(kHost)),
      return_index ? out_index->mutable_data<int>(TARGET(kHost)) : nullptr,
      &batch_starts,
      n,
      box_num,
      class_num,
      out_dim,
      std::min(param.nms_top_k, 128),
      param.score_threshold,
      std::min(param.keep_top_k, 100),
      param.nms_threshold,
      param.background_label,
      param.normalized,
      param.nms_eta,
      return_index);

  CHECK_EQ(r, 0);

  uint64_t num_kept = batch_starts.back();
  if (num_kept == 0) {
      if (return_index) {
          outs->Resize({0, out_dim});
          out_index->Resize({0, 1});
      } else {
          outs->Resize({1, 1});
          float* od = outs->mutable_data<float>(TARGET(kHost));
          od[0] = -1;
          batch_starts = {0, 1};
      }
  } else {
      outs->Resize({static_cast<int64_t>(num_kept), out_dim});
      if (return_index) {
          out_index->Resize({static_cast<int64_t>(num_kept), 1});
      }
  }

  LoD lod;
  lod.emplace_back(batch_starts);
  if (return_index) {
      out_index->set_lod(lod);
  }
  outs->set_lod(lod);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(multiclass_nms,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::MulticlassNmsCompute,
                     def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(multiclass_nms2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::MulticlassNmsCompute,
                     def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Index", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
