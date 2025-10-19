from .detector3d_template import Detector3DTemplate


class PVRCNNPlusPlus(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        batch_dict = self.roi_head.proposal_layer(
            batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
            
    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True
                roi_labels: (B, num_rois)
                roi_scores: (B, num_rois)
                rois: (B, num_rois, 7)

        Returns:
            pred_dicts: list of pred_dicts
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING # 기본 설정을 먼저 가져옴
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):

            record_dict = {
                'pred_boxes': batch_dict['rois'][index],
                'pred_scores': batch_dict['roi_scores'][index],
                'pred_labels': batch_dict['roi_labels'][index]
            }

            # 수정
            roi_head_test_cfg = self.model_cfg.ROI_HEAD.NMS_CONFIG.TEST
            score_thresh_cfg = roi_head_test_cfg.SCORE_THRESH

            # 클래스별 임계값 적용 로직
            total_mask = torch.zeros_like(record_dict['pred_scores'], dtype=torch.bool)

            if isinstance(score_thresh_cfg, dict):
                # 클래스 이름은 self.class_names (['Vehicle', 'Pedestrian', 'Cyclist'])를 사용
                # pred_labels는 1부터 시작하므로 인덱스 처리에 주의 (label 1 -> class_names[0])
                for i, class_name in enumerate(self.class_names):
                    class_thresh = score_thresh_cfg.get(class_name, 0.1) # yaml에 없으면 0.1

                    # 현재 클래스에 해당하고, 점수가 임계값보다 높은 경우만 True로 설정
                    class_mask = (record_dict['pred_labels'] == (i + 1)) & (record_dict['pred_scores'] > class_thresh)
                    total_mask = total_mask | class_mask
            else:
                # 단일 값일 경우
                total_mask = record_dict['pred_scores'] > score_thresh_cfg

            selected = torch.where(total_mask)[0]

            record_dict['pred_boxes'] = record_dict['pred_boxes'][selected]
            record_dict['pred_scores'] = record_dict['pred_scores'][selected]
            record_dict['pred_labels'] = record_dict['pred_labels'][selected]

            # NMS 적용
            selected, _ = model_nms_utils.class_agnostic_nms(
                box_scores=record_dict['pred_scores'], box_preds=record_dict['pred_boxes'],
                nms_config=roi_head_test_cfg, # NMS 설정도 ROI_HEAD의 것을 사용
            )

            final_boxes = record_dict['pred_boxes'][selected]
            final_scores = record_dict['pred_scores'][selected]
            final_labels = record_dict['pred_labels'][selected]

            pred_dict = {
                'pred_boxes': final_boxes.cpu().numpy(),
                'pred_scores': final_scores.cpu().numpy(),
                'pred_labels': final_labels.cpu().numpy(),
                'pred_names': np.array([self.class_names[x - 1] for x in final_labels.cpu().numpy()])
            }

            pred_dicts.append(pred_dict)

        return pred_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
