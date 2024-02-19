from transformers import BeamSearchScorer
from abc import ABC, abstractmethod
from collections import UserDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

class BeamSearchScorerConstrained(BeamSearchScorer):
    #Beam_search constraints are added, that alter the score of a beam according to the constraints, they are given as a list of objects from the class Constraint
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
        constraints = [],
    ):

        super().__init__(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=device,
            length_penalty=length_penalty,
            do_early_stopping=do_early_stopping,
            num_beam_hyps_to_keep=num_beam_hyps_to_keep,
            num_beam_groups=num_beam_groups,
        )

        self.constraints = constraints
    

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        # add up to the length which the next_scores is calculated on
        cur_len = input_ids.shape[-1] - decoder_prompt_len + 1
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for batch_idx in range(batch_size):
            batch_group_idx = batch_idx * self.num_beam_groups + group_index
            if self._done[batch_group_idx]:
                if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index


                for constraint in self.constraints:
                    next_score = constraint.apply_beam_constraint(next_token, next_score, input_ids[batch_beam_idx], cur_len, self.length_penalty)
                

                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (batch_beam_idx,)
                    else:
                        beam_index = None

                    # skip the corner case where the very first generated token is eos_token
                    if decoder_prompt_len == input_ids.shape[-1]:
                        continue
                    

                    self._beam_hyps[batch_group_idx].add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                        decoder_prompt_len=decoder_prompt_len,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )
    
