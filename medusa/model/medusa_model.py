import copy
import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
# import transformers
import pdb
# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

class MedusaHead(nn.Module):
    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__()

        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = config.base_model_name_or_path
        # Create a list of Medusa heads
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )

    def get_medusa_choice(self, model_name):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        warnings.warn('Please specify medusa choice configuration!')
        return mc_sim_7b_63
    
    def forward(
        self,
        hidden_states,
        **kwargs,
    ):
        # Clone the output hidden states
        hidden_states = hidden_states.clone()
        medusa_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa_num_heads):
            medusa_logits.append(self.heads[i](hidden_states))
        return torch.stack(medusa_logits, dim=0)

    def generate(
        self,
        input_ids,
        prompt_valid_length=None,
        temperature=0.0,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True,
        hidden_states=None,
        logits=None,
        pad_token_id=0,
        prompt_run=False,
        candidates=None,
    ):
        medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=input_ids.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices
        if prompt_run:
            # Cache medusa buffers (the fixed patterns for tree attention)

            medusa_logits = self.forward(hidden_states)
            last_logits = extract_last_valid_logits(logits, prompt_valid_length)
            last_medusa_logits = extract_last_valid_logits(medusa_logits, prompt_valid_length)
            new_ids = None
            select_indices = None
            valid_length = prompt_valid_length
        else:
            tree_medusa_logits = self.forward(hidden_states)
            tree_logits = logits
            # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
            logits = tree_logits[:, medusa_buffers["retrieve_indices"]]
            medusa_logits = tree_medusa_logits[:, :, medusa_buffers["retrieve_indices"]]

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )
            # Update the logits
            # if 0 prediction is accepted, we have one new token
            # because the orginal output of base model is the always correct
            valid_length = accept_length + 1
            new_ids, select_indices, last_logits, last_medusa_logits = get_new_ids(
                candidates,
                medusa_buffers["retrieve_indices"],
                best_candidate,
                valid_length,
                logits,
                medusa_logits,
                padding_token_id=pad_token_id
            )
            select_indices+=input_ids.shape[1]

        # Generate candidates with topk predictions from Medusa heads
        candidates, tree_candidates = generate_candidates(
            last_medusa_logits,
            last_logits,
            medusa_buffers["tree_indices"],
            medusa_buffers["retrieve_indices"],
            temperature=temperature,
            posterior_alpha=posterior_alpha,
            posterior_threshold=posterior_threshold,
            top_p=top_p,
            sampling=sampling,
            fast=fast,
        )
        cross_step_params = {
            "tree_candidates": tree_candidates,
            "medusa_position_ids": medusa_buffers["medusa_position_ids"],
            "candidates": candidates,
            "tree_candidates": tree_candidates,
            "medusa_attn_mask": medusa_buffers["medusa_attn_mask"],
            "new_ids": new_ids,
            "select_indices": select_indices,
        }
        return cross_step_params, valid_length


class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs
        self.medusa_head = MedusaHead(config)

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
        # Create a list of Medusa heads
    # Add a link named base_model to self
    @property
    def base_model(self):
        return self

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except:
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = 5 # TODO: fix the uploaded config (only include 2 heads)
            base_model_config.medusa_num_layers = config.medusa_num_layers
            base_model_config.base_model_name_or_path = config.base_model_name_or_path
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
            medusa_head_state_dict = torch.load(filename, map_location=model.device)
            for key in list(medusa_head_state_dict.keys()):
                medusa_head_state_dict["heads." + key] = medusa_head_state_dict.pop(key)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        **kwargs,
    ):
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        if output_orig:
            return outputs, orig
        else:
            return outputs


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        if output_orig:
            outputs, orig = self.model_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_orig=True,
                position_ids=position_ids,
                **kwargs,
            )
        else:
            outputs = self.model_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        medusa_logits = self.medusa_head_forward(outputs[0])
        if output_orig:
            return medusa_logits, outputs, orig
        return medusa_logits



    def medusa_generate(
        self,
        input_ids,
        input_id_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            input_id_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """

        hyper_params = {
            "temperature": temperature,
            "posterior_threshold": posterior_threshold,
            "posterior_alpha": posterior_alpha,
            "top_p": top_p,
            "sampling": sampling,
            "fast": fast,
        }
        batch_size = input_ids.shape[0]
        # Initialize the past key and value states
        if hasattr(self, "past_key_values") and batch_size==self.past_key_values_data.shape[1]:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, batch_size)
        self.past_key_values = past_key_values
        self.past_key_values_data = past_key_values_data
        self.current_length_data = current_length_data

        self.base_model.model.medusa_mask = None
        self.base_model.model.medusa_mode = None
        # Initialize tree attention mask and process prefill tokens
        outputs, logits = self.model_forward(
            input_ids,
            attention_mask=input_id_mask,
            past_key_values=past_key_values,
            output_orig=True,
        )
        cross_step_params, valid_length = self.medusa_head.generate(
            input_ids,
            prompt_valid_length=input_id_mask.sum(dim=1),
            hidden_states=outputs[0],
            logits=logits,
            prompt_run=True,
            **hyper_params,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Use tree attention to verify the candidates and get predictions
        # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence.
        position_ids_with_medusa = update_position_id(cross_step_params["medusa_position_ids"], input_id_mask, input_ids)
        input_id_mask_with_medusa = update_attention_mask(input_id_mask, cross_step_params["tree_candidates"])
        self.base_model.model.medusa_mask = cross_step_params["medusa_attn_mask"]
        input_len = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
        if isinstance(input_len, int):
            ends = [input_len] * batch_size
        else:
            ends = copy.deepcopy(input_len)

        target_length = torch.ones(batch_size, dtype=torch.int, device=input_ids.device)*max_steps
        any_finished = torch.any(target_length<=0)
        all_finished = torch.all(target_length<=0)
        while not all_finished:
            # Use the model to decode the tree candidates. 
            # The model is expected to return logits for the Medusa structure, original logits, and possibly other outputs.
            print('start decode')
            outputs, tree_logits = self.model_forward(
                cross_step_params["tree_candidates"],
                past_key_values=past_key_values,
                output_orig=True,
                position_ids=position_ids_with_medusa,
                attention_mask=input_id_mask_with_medusa,
            )
            print('decode input:', cross_step_params["tree_candidates"])
            print('decode logits:', outputs[0])
            import pdb; pdb.set_trace()
            cross_step_params, valid_length = self.medusa_head.generate(
                input_ids,
                prompt_valid_length=None,
                **hyper_params,
                hidden_states=outputs[0],
                prompt_run=False,
                logits=tree_logits,
                candidates=cross_step_params["candidates"],
                pad_token_id=self.tokenizer.pad_token_id,
            )
            input_ids, scatter_index = update_ids_new(input_ids, cross_step_params["new_ids"], padding_value=self.tokenizer.pad_token_id)
            tgt = gather_from_past_key_values(self.past_key_values_data, cross_step_params["select_indices"])
            update_kvcache_new(tgt, self.past_key_values_data, scatter_index)
            self.current_length_data.fill_(input_ids.shape[-1])
            input_id_mask = update_mask_new(input_id_mask, valid_length)

            # Use tree attention to verify the candidates and get predictions
            # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence.
            position_ids_with_medusa = update_position_id(cross_step_params["medusa_position_ids"], input_id_mask, input_ids)
            input_id_mask_with_medusa = update_attention_mask(input_id_mask, cross_step_params["tree_candidates"])

            decoded_texts = []
            eos_encountered = [False] * batch_size
            target_length -= valid_length
            any_finished = torch.any(target_length<=0)
            all_finished = torch.all(target_length<=0)
            for i in range(batch_size):
                if isinstance(input_len, int):
                    input_len_ = input_len
                else:
                    input_len_ = input_len[i]
                # 检查当前批次的文本是否包含结束符
                if self.tokenizer.eos_token_id in input_ids[i, input_len_:]:
                    eos_encountered[i] = True
                else:
                    ends[i] = len(input_ids[i])
                decoded_text = self.tokenizer.decode(
                    input_ids[i, input_len_:ends[i]],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                decoded_texts.append(decoded_text)
            yield{ "text": decoded_texts}

            # 如果所有批次都遇到了 EOS，则停止
            if all(eos_encountered):
                break


class MedusaModelLlama(MedusaModelABC, KVLlamaForCausalLM):
    pass

class MedusaModelMistral(MedusaModelABC, KVMistralForCausalLM):
    pass


class MedusaModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return MedusaModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return MedusaModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")
