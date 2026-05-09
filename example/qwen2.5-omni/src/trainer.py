# ruff: noqa: UP045

from typing import Optional

import torch
from transformers import Trainer
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM


from edgerazor import EdgeRazor
from edgerazor.kd.util.moe_loss import router_z_loss_func


# EdgeRazor API: Multi-level Online Knowledge Distillation for Dense / Mixture-of-Experts Models
class EdgeRazorTrainer(Trainer):
    def __init__(
            self,
            model=None,
            args=None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            preprocess_logits_for_metrics=None,
            # Distill Configuration
            teacher_model=None,
            router_aux_loss_coef=0.01,
            router_z_loss_coef=0.001,
            edgerazor: EdgeRazor=None,
            use_audio_in_video: bool=False,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model.thinker
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.edgerazor = edgerazor
        self.use_audio_in_video = use_audio_in_video
        
        self.custom_losses = {}  # Store custom losses
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs contains: input_ids, attention_mask, labels
        model = model.thinker  # Student thinker model

        # Keep all in student device
        student_device = next(model.parameters()).device
        self.teacher_model.to(student_device)
        for key, value in inputs.items():
            inputs[key] = value.to(student_device)

        # Student model inference
        student_outputs = model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
            use_audio_in_video=self.use_audio_in_video,  # For Qwen2.5-Omni
            # output_attentions=True,
            # output_past_key_values=True,
        )
        
        # Teacher model inference
        with torch.no_grad():
            teacher_device = next(self.teacher_model.parameters()).device
            # Create a copy of input data and move to teacher model device
            teacher_inputs = {}
            for key, value in inputs.items():
                teacher_inputs[key] = value.to(teacher_device)
            teacher_outputs = self.teacher_model(
                **teacher_inputs,
                return_dict=True,
                output_hidden_states=True,
                use_audio_in_video=self.use_audio_in_video,  # For Qwen2.5-Omni
                # output_attentions=True,
                # output_past_key_values=True,
            )

        labels = inputs['labels'].to(student_device)

        if student_outputs.logits.shape[-1] != teacher_outputs.logits.shape[-1]:
            raise ValueError("Student and teacher logits must have same shape!")

        # # DEBUG
        # print(f"labels shape: {labels.shape}") # labels shape: torch.Size([2, 4096])
        # print(f"teacher_logits shape: {teacher_logits.shape}") # teacher_logits shape: torch.Size([2=batch_size, 4096=context_len, 50304=vocab_size])

        # Calculate loss
        loss_total, loss_dict = self.edgerazor.compute_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            labels=labels,
        )
        loss_task = loss_dict.get('task_loss', 0.0)
        loss_dist = loss_dict.get('distill_loss', 0.0)

        # MoE loss
        if isinstance(model, (OlmoeForCausalLM, )):
            student_router_logits = torch.stack(list(student_outputs.router_logits), dim=0)
            aux_loss = self.router_aux_loss_coef * student_outputs.aux_loss # from load_balancing_loss_func
            router_z_loss = self.router_z_loss_coef * router_z_loss_func(student_router_logits)
            loss_total = loss_total + aux_loss + router_z_loss
        
        # # DEBUG
        # print(f"labels shape: {labels.shape}") # torch.Size([2, 4096])
        # print(f"student_router_logits shape: {student_router_logits.shape}") # torch.Size([16, 8192, 64])
        # print(f"teacher_router_logits shape: {teacher_router_logits.shape}") # torch.Size([16=num_hidden_layers, 8192=2*4096, 64=num_experts])

        # LOG
        # Store custom losses
        self.custom_losses = {
            'train/loss_total': loss_total.item() if isinstance(loss_total, torch.Tensor) else loss_total,
            # EdgeRazor losses
            'train/loss_task': loss_task.item() if isinstance(loss_task, torch.Tensor) else loss_task,
            'train/loss_dist': loss_dist.item() if isinstance(loss_dist, torch.Tensor) else loss_dist,
        }
        # Update KD detailed losses: train/loss_dist_*
        for key, value in loss_dict.get('distill_loss_details', {}).items():
            # key name is usually like 'loss_1', 'loss_2', etc.
            ind = key[5:]
            self.custom_losses[f'train/loss_dist_{ind}'] = value.item() if isinstance(value, torch.Tensor) else value
        
        # Update MoE losses
        if isinstance(model, (OlmoeForCausalLM, )):
            loss_gate_kld = loss_dict.get('distill_loss_details', {}).get('loss_gate_kld', 0.0)
            self.custom_losses.update({
                'train/loss_gate_kld': loss_gate_kld.item() if isinstance(loss_gate_kld, torch.Tensor) else loss_gate_kld,
                'train/aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
                'train/router_z_loss': router_z_loss.item() if isinstance(router_z_loss, torch.Tensor) else router_z_loss,
            })
        
        return (loss_total, student_outputs) if return_outputs else loss_total
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Add custom losses to logs
        logs.update(self.custom_losses)
        # Call parent class log method
        super().log(logs, start_time)
